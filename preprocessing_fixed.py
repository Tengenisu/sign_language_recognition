import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import logging
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration matching CorrFormer-Lite paper specifications
CONFIG = {
    "data_root": "autsl/",
    "out_root": "processed_autsl/",
    "splits": ["train", "val", "test"],
    "max_frames": 64,  # Fixed 64 frames as per paper (Section III-A)
    
    # MediaPipe models
    "pose_model": "pose_landmarker_lite.task",
    "hand_model": "hand_landmarker.task",
    
    # Processing parameters
    "num_workers": 6,  # Reduced for stability
    "min_confidence": 0.3,
    "skip_empty_threshold": 0.95,  # Increased from 0.8 - more lenient
    "use_gpu": False,
    "use_video_mode": True,  # If False, uses IMAGE mode (slower but more reliable)
    
    # Keypoint configuration (56 total as per paper Section IV-D)
    # 14 upper-body pose + 21 left hand + 21 right hand = 56 landmarks
    "num_pose_landmarks": 14,
    "num_hand_landmarks": 21,
    "total_landmarks": 56,
}

# Pose landmark indices for upper body (14 landmarks)
# Selected for sign language: head, shoulders, arms, torso
UPPER_BODY_INDICES = [
    0,        # nose
    11, 12,   # left shoulder, right shoulder
    13, 14,   # left elbow, right elbow
    15, 16,   # left wrist, right wrist
    23, 24,   # left hip, right hip
    7, 8,     # left ear, right ear
    9, 10,    # left eye, right eye
]

GLOBAL_POSE = None
GLOBAL_HAND = None


def init_worker(pose_model, hand_model, use_gpu=False, use_video_mode=True):
    """
    Initialize pose and hand detectors ONCE per worker.
    """
    global GLOBAL_POSE, GLOBAL_HAND

    # Silence stderr in the worker
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass

    try:
        # Determine running mode
        running_mode = vision.RunningMode.VIDEO if use_video_mode else vision.RunningMode.IMAGE
        
        # Pose Landmarker
        base_opts_p = python.BaseOptions(model_asset_path=pose_model)
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=base_opts_p,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.1,
            min_pose_presence_confidence=0.1,
            min_tracking_confidence=0.1
        )
        GLOBAL_POSE = vision.PoseLandmarker.create_from_options(pose_opts)

        # Hand Landmarker
        base_opts_h = python.BaseOptions(model_asset_path=hand_model)
        hand_opts = vision.HandLandmarkerOptions(
            base_options=base_opts_h,
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1
        )
        GLOBAL_HAND = vision.HandLandmarker.create_from_options(hand_opts)

        return True

    except Exception as e:
        with open("worker_init_error.log", "a") as f:
            f.write(f"Worker {os.getpid()} initialization failed:\n")
            f.write(f"{type(e).__name__}: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc() + "\n\n")
        return False


def extract_upper_body_landmarks(pose_landmarks):
    """
    Extract 14 upper body landmarks from 33 pose landmarks.
    Returns: (14, 3) array with (x, y, z)
    """
    upper_body = np.zeros((14, 3), dtype=np.float32)
    
    for i, idx in enumerate(UPPER_BODY_INDICES):
        if idx < len(pose_landmarks):
            lm = pose_landmarks[idx]
            upper_body[i] = [lm.x, lm.y, lm.z]
    
    return upper_body


def build_frame_landmarks(pose_res, hand_res) -> tuple:
    """
    Convert MediaPipe results to a (56, 3) array as per paper:
    14 upper-body pose + 21 left hand + 21 right hand = 56 landmarks
    Returns: (frame_landmarks, has_detection)
    
    Paper Section IV-D: "56 keypoints per frame—14 upper-body joints 
    and 21 each for the left and right hands"
    """
    frame_landmarks = np.zeros((56, 3), dtype=np.float32)
    has_detection = False
    detected_parts = []

    # Extract upper body pose (14 landmarks) -> indices 0-13
    if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
        upper_body = extract_upper_body_landmarks(pose_res.pose_landmarks[0])
        frame_landmarks[0:14] = upper_body
        has_detection = True
        detected_parts.append("pose")

    # Extract hand landmarks
    # Left hand: indices 14-34 (21 landmarks)
    # Right hand: indices 35-55 (21 landmarks)
    if hand_res and hand_res.hand_landmarks and hand_res.handedness:
        for hand_idx, hand_lms in enumerate(hand_res.hand_landmarks):
            if hand_idx >= len(hand_res.handedness):
                continue
                
            handedness = hand_res.handedness[hand_idx][0].category_name
            
            if handedness == "Left":
                start_idx = 14
                for i, lm in enumerate(hand_lms[:21]):  # Ensure only 21
                    frame_landmarks[start_idx + i] = [lm.x, lm.y, lm.z]
                detected_parts.append("left_hand")
            else:  # Right
                start_idx = 35
                for i, lm in enumerate(hand_lms[:21]):  # Ensure only 21
                    frame_landmarks[start_idx + i] = [lm.x, lm.y, lm.z]
                detected_parts.append("right_hand")
            
            has_detection = True

    return frame_landmarks, has_detection


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize sequence as per paper Equations (2)-(4):
    - Center by mean
    - Scale by standard deviation
    
    Input: (T, 56, 3)
    Output: (T, 56, 3) normalized
    """
    T, J, D = sequence.shape
    normalized = np.zeros_like(sequence)
    
    for t in range(T):
        frame = sequence[t]  # (56, 3)
        
        # Compute mean μ_t (Equation 3)
        mu_t = np.mean(frame, axis=0)  # (3,)
        
        # Compute std σ_t (Equation 4)
        centered = frame - mu_t
        sigma_t = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        
        # Avoid division by zero
        if sigma_t < 1e-6:
            sigma_t = 1.0
        
        # Apply normalization (Equation 2)
        normalized[t] = (frame - mu_t) / sigma_t
    
    return normalized


def find_video_file(base_path: Path, video_name: str, split: str):
    """
    Try common AUTSL patterns: vid_color.mp4, vid.mp4, etc.
    """
    path = base_path / split / f"{video_name}_color.mp4"
    if path.exists():
        return path

    # Fallbacks
    exts = [".mp4", ".avi", ".MOV"]
    suffixes = ["_color", ""]
    for ext in exts:
        for suf in suffixes:
            p = base_path / split / f"{video_name}{suf}{ext}"
            if p.exists():
                return p
    return None


def process_single_video(task):
    """
    Process ONE video:
    - Read ALL frames from video
    - Extract landmarks for each frame
    - Uniformly sample to 64 frames
    - Normalize using paper's method
    - Output shape: (64, 56, 3)
    """
    video_path, label, video_name, config = task

    global GLOBAL_POSE, GLOBAL_HAND

    if GLOBAL_POSE is None or GLOBAL_HAND is None:
        return {"error": "Detectors not initialized", "vid": video_name}

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}", "vid": video_name}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or total_frames is None:
            # Fallback: count frames manually
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.release()
            
            if total_frames == 0:
                return {"error": "Video has 0 frames", "vid": video_name}
            
            # Re-open video
            cap = cv2.VideoCapture(str(video_path))
        
        if fps <= 0 or fps is None:
            fps = 25.0

        # Process ALL frames first, then sample
        all_landmarks = []
        empty_count = 0
        frame_idx = 0
        frames_processed = 0
        detection_debug = {"pose": 0, "left_hand": 0, "right_hand": 0, "neither": 0}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # CRITICAL: timestamp must be in milliseconds and monotonically increasing
            timestamp_ms = int((frame_idx * 1000) / fps)

            try:
                # Choose detection method based on mode
                if config.get("use_video_mode", True):
                    # VIDEO mode - requires monotonic timestamps
                    pose_res = GLOBAL_POSE.detect_for_video(mp_img, timestamp_ms)
                    hand_res = GLOBAL_HAND.detect_for_video(mp_img, timestamp_ms)
                else:
                    # IMAGE mode - more reliable but slower
                    pose_res = GLOBAL_POSE.detect(mp_img)
                    hand_res = GLOBAL_HAND.detect(mp_img)

                # Build landmarks from results
                frame_landmarks, has_det = build_frame_landmarks(pose_res, hand_res)
                all_landmarks.append(frame_landmarks)

                # Track what was detected for debugging
                if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                    detection_debug["pose"] += 1
                if hand_res and hand_res.hand_landmarks:
                    for hand_idx, _ in enumerate(hand_res.hand_landmarks):
                        if hand_idx < len(hand_res.handedness):
                            handedness = hand_res.handedness[hand_idx][0].category_name
                            if handedness == "Left":
                                detection_debug["left_hand"] += 1
                            else:
                                detection_debug["right_hand"] += 1
                
                if not has_det:
                    empty_count += 1
                    detection_debug["neither"] += 1

            except Exception as e:
                # If detection fails, push zeros
                frame_landmarks = np.zeros((56, 3), dtype=np.float32)
                all_landmarks.append(frame_landmarks)
                empty_count += 1
                detection_debug["neither"] += 1

            frame_idx += 1

        cap.release()
        
        # Debug: verify we actually processed frames
        if frames_processed != len(all_landmarks):
            return {
                "error": f"Frame mismatch: processed {frames_processed}, extracted {len(all_landmarks)}", 
                "vid": video_name
            }

        # Check if we have enough valid frames
        if not all_landmarks or len(all_landmarks) < 5:
            return {"error": f"Too few frames extracted: {len(all_landmarks)}", "vid": video_name}

        # Calculate detection rate
        detection_rate = (frames_processed - empty_count) / frames_processed if frames_processed > 0 else 0
        empty_ratio = empty_count / len(all_landmarks)
        
        # More lenient check: allow if we have ANY detections
        if empty_ratio > config["skip_empty_threshold"]:
            return {
                "error": f"Too many empty frames: {empty_ratio:.2%} (detected in {(1-empty_ratio)*100:.1f}% of frames)", 
                "vid": video_name,
                "frames_processed": frames_processed,
                "empty_count": empty_count
            }

        # Convert to array: (T, 56, 3)
        full_seq = np.array(all_landmarks, dtype=np.float32)
        
        # Uniformly sample to exactly 64 frames
        max_frames = config["max_frames"]
        if len(full_seq) > max_frames:
            # Uniform sampling using linspace
            indices = np.linspace(0, len(full_seq) - 1, max_frames, dtype=int)
            seq = full_seq[indices]
        elif len(full_seq) < max_frames:
            # Pad with zeros
            pad = np.zeros((max_frames - len(full_seq), 56, 3), dtype=np.float32)
            seq = np.concatenate([full_seq, pad], axis=0)
        else:
            seq = full_seq

        # Normalize sequence (Equations 2-4 from paper)
        seq = normalize_sequence(seq)

        return {
            "vid": video_name,
            "lbl": label,
            "data": seq,  # Shape: (64, 56, 3)
            "original_frames": len(all_landmarks),
            "empty_frames": empty_count,
            "detection_debug": detection_debug,
            "success": True,
        }

    except Exception as e:
        import traceback
        return {
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
            "vid": video_name,
        }


def process_split(split: str, config: dict):
    base_path = Path(config["data_root"])
    out_root = Path(config["out_root"])

    print(f"\n{'='*80}")
    print(f"📦 Processing: {split.upper()}")
    print(f"{'='*80}")

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_path / f"{split}_labels.csv"
    if not csv_path.exists():
        print(f"⚠️  CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, header=None, names=["vid", "lbl"])
    print(f"📋 Found {len(df)} entries in CSV")

    tasks = []
    missing = 0
    for _, row in df.iterrows():
        vid = str(row["vid"])
        vp = find_video_file(base_path, vid, split)
        if vp:
            tasks.append((str(vp), int(row["lbl"]), vid, config))
        else:
            missing += 1

    print(f"✅ Videos found: {len(tasks)}/{len(df)}")
    if missing > 0:
        print(f"⚠️  Missing videos: {missing}")

    if not tasks:
        print("❌ No videos to process for this split.")
        return

    metadata = []
    successful = 0
    failed = 0
    error_summary = {}
    error_examples = {}
    
    start_time = time.time()

    print(f"🔥 Starting processing with {config['num_workers']} workers...")
    print(f"⏳ This will take some time - processing {len(tasks)} videos...")

    with ProcessPoolExecutor(
        max_workers=config['num_workers'],
        initializer=init_worker,
        initargs=(config["pose_model"], config["hand_model"], config["use_gpu"], config.get("use_video_mode", True))
    ) as executor:
        futures = {executor.submit(process_single_video, t): t for t in tasks}

        pbar = tqdm(
            total=len(tasks),
            desc=f"{split:5}",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result and result.get("success"):
                    save_path = out_dir / f"{result['vid']}.npz"
                    np.savez_compressed(
                        save_path, 
                        data=result["data"],  # (64, 56, 3)
                        label=result["lbl"]
                    )

                    metadata.append({
                        "video": result["vid"],
                        "path": str(save_path.relative_to(config["out_root"])),
                        "label": result["lbl"],
                        "frames": result["original_frames"],
                        "empty_frames": result.get("empty_frames", 0),
                        "shape": str(result["data"].shape)
                    })
                    successful += 1
                    
                    # Debug: print first few results
                    if successful <= 3:
                        print(f"\n[DEBUG] Video: {result['vid']}, Frames: {result['original_frames']}, Empty: {result.get('empty_frames', 0)}, Shape: {result['data'].shape}")
                else:
                    failed += 1
                    if result and "error" in result:
                        err = result["error"]
                        error_summary[err] = error_summary.get(err, 0) + 1
                        if err not in error_examples:
                            error_examples[err] = {
                                "vid": result.get("vid", "unknown"),
                                "traceback": result.get("traceback", "")
                            }
            except Exception as e:
                failed += 1
                err = f"Future exception: {type(e).__name__}: {str(e)}"
                error_summary[err] = error_summary.get(err, 0) + 1

            pbar.update(1)

        pbar.close()

    elapsed = time.time() - start_time
    
    if metadata:
        meta_df = pd.DataFrame(metadata)
        meta_path = out_root / f"{split}_meta.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"💾 Metadata saved: {meta_path}")
        print(f"📊 Output shape per sample: (64, 56, 3)")
        
        # Statistics
        avg_frames = meta_df['frames'].mean()
        avg_empty = meta_df['empty_frames'].mean()
        print(f"📈 Average original frames per video: {avg_frames:.1f}")
        print(f"📉 Average empty frames per video: {avg_empty:.1f}")

    total = successful + failed if (successful + failed) > 0 else 1
    print(f"✅ Success: {successful:,} | ❌ Failed: {failed:,} | Success rate: {successful/total*100:.1f}%")
    print(f"⏱️  Processing time: {elapsed/60:.1f} minutes ({elapsed/successful:.2f}s per video)")

    if error_summary:
        print("\n🔍 Error breakdown (top 5):")
        for err, count in sorted(error_summary.items(), key=lambda x: -x[1])[:5]:
            print(f"   [{count:,}] {err}")
            if err in error_examples:
                ex = error_examples[err]
                print(f"      Example: {ex['vid']}")


def main():
    # Check models
    if not Path(CONFIG["pose_model"]).exists():
        print(f"❌ Missing pose model: {CONFIG['pose_model']}")
        print("   Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        return
    if not Path(CONFIG["hand_model"]).exists():
        print(f"❌ Missing hand model: {CONFIG['hand_model']}")
        print("   Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    print("="*80)
    print("🚀 AUTSL PREPROCESSING - CorrFormer-Lite Paper Specifications")
    print("="*80)
    print(f"📄 Paper: CorrFormer-Lite (Vohra et al.)")
    print(f"💻 Workers: {CONFIG['num_workers']}")
    print(f"🎮 GPU: {'Enabled' if CONFIG['use_gpu'] else 'Disabled (CPU only)'}")
    print(f"📊 Output format: (64, 56, 3) per sample")
    print(f"   - 64 frames (uniform sampling)")
    print(f"   - 56 landmarks (14 upper-body + 21 left hand + 21 right hand)")
    print(f"   - 3 coordinates (x, y, z)")
    print(f"   - Detection threshold: {CONFIG['skip_empty_threshold']*100:.0f}% (relaxed)")
    print(f"   - Min confidence: lowered to 0.1 for better detection")
    print(f"📂 Data root: {CONFIG['data_root']}")
    print(f"📂 Output root: {CONFIG['out_root']}")
    print("="*80)
    
    # Test one video first
    print("\n🧪 Testing detection on one video first...")
    test_csv = Path(CONFIG["data_root"]) / "train_labels.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv, header=None, names=["vid", "lbl"])
        test_vid = str(df.iloc[0]["vid"])
        test_path = find_video_file(Path(CONFIG["data_root"]), test_vid, "train")
        if test_path:
            print(f"   Testing: {test_vid}")
            print(f"   Path: {test_path}")
            
            # Initialize detectors for test
            init_worker(CONFIG["pose_model"], CONFIG["hand_model"], CONFIG["use_gpu"], CONFIG.get("use_video_mode", True))
            
            result = process_single_video((str(test_path), 0, test_vid, CONFIG))
            if result.get("success"):
                print(f"   ✅ SUCCESS!")
                print(f"      Total frames: {result['original_frames']}")
                print(f"      Empty frames: {result['empty_frames']}")
                debug = result.get('detection_debug', {})
                print(f"      Pose detected: {debug.get('pose', 0)} frames")
                print(f"      Left hand: {debug.get('left_hand', 0)} frames")
                print(f"      Right hand: {debug.get('right_hand', 0)} frames")
                print(f"      Neither: {debug.get('neither', 0)} frames")
            else:
                print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                if "frames_processed" in result:
                    print(f"      Processed: {result['frames_processed']} frames")
                    print(f"      Empty: {result['empty_count']} frames")
                if "detection_debug" in result:
                    debug = result['detection_debug']
                    print(f"      Pose detected: {debug.get('pose', 0)} frames")
                    print(f"      Left hand: {debug.get('left_hand', 0)} frames")
                    print(f"      Right hand: {debug.get('right_hand', 0)} frames")
            
            print("\n❓ Continue with full processing? (Press Ctrl+C to cancel)")
            import time
            time.sleep(3)
            print()

    # Clear old worker error log
    if Path("worker_init_error.log").exists():
        Path("worker_init_error.log").unlink()

    overall_start = time.time()
    
    for split in CONFIG["splits"]:
        process_split(split, CONFIG)

    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*80)
    print("🎉 PREPROCESSING COMPLETE!")
    print("="*80)

    out_root = Path(CONFIG["out_root"])
    print("📊 Files per split:")
    total_samples = 0
    for split in CONFIG["splits"]:
        d = out_root / split
        if d.exists():
            count = len(list(d.glob("*.npz")))
            total_samples += count
            print(f"   {split:5}: {count:,} NPZ files")
        else:
            print(f"   {split:5}: Directory not found")
    
    print(f"\n📦 Total samples processed: {total_samples:,}")
    print(f"⏱️  Total processing time: {overall_elapsed/3600:.2f} hours")
    print(f"💾 Each .npz file contains:")
    print(f"   - 'data': shape (64, 56, 3) - normalized skeleton sequence")
    print(f"   - 'label': integer class label (0-225 for AUTSL)")

    if Path("worker_init_error.log").exists():
        print("\n⚠️  Worker init errors detected. See worker_init_error.log")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()