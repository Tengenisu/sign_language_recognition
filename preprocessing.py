"""
AUTSL Dataset Preprocessing - Ultra-Optimized Version
Uses threading + MediaPipe to avoid multiprocessing crashes
Expected: ~3-5 hours for full dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gc
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

CONFIG = {
    # Dataset paths
    'data_root': '/path/to/AUTSL',  # CHANGE THIS
    'output_root': 'data/processed_fast',
    
    # Processing settings
    'splits': ['train', 'val', 'test'],
    'max_frames': 64,
    'normalize': True,
    
    # Performance tuning
    'num_workers': 4,  # Concurrent MediaPipe instances (4-6 recommended)
    'use_threading': True,  # Use threading instead of multiprocessing (safer for MediaPipe)
    'chunk_size': 100,  # Process videos in chunks to save progress
    
    # MediaPipe settings
    'model_complexity': 0,  # 0=fastest, 1=balanced, 2=accurate
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.3,
}

# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def extract_video_frames_fast(video_path, max_frames=None):
    """Optimized frame extraction with sampling"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    # Calculate indices for uniform sampling
    if max_frames and total_frames > max_frames:
        stride = total_frames / max_frames
        frame_indices = set([int(i * stride) for i in range(max_frames)])
    else:
        frame_indices = set(range(total_frames))
    
    frames = []
    current_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_idx in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        current_idx += 1
        
        if len(frames) >= (max_frames or float('inf')):
            break
    
    cap.release()
    return frames if len(frames) > 0 else None


def extract_keypoints_from_frame(results, pose_indices):
    """Fast keypoint extraction"""
    keypoints = []
    
    # Pose (upper body only)
    if results.pose_landmarks:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.visibility])
    else:
        keypoints.extend([0.0] * (len(pose_indices) * 3))
    
    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 63)  # 21 * 3
    
    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 63)
    
    return np.array(keypoints, dtype=np.float32).reshape(-1, 3)


def normalize_sequence_fast(sequence):
    """Fast numpy-based normalization"""
    T, J, D = sequence.shape
    normalized = np.copy(sequence)
    
    coords = normalized[:, :, :2]
    confidence = normalized[:, :, 2:3]
    
    for t in range(T):
        frame_coords = coords[t]
        frame_conf = confidence[t]
        
        # Center on reference point
        valid_mask = frame_conf.squeeze() > 0.1
        if valid_mask.any():
            ref_point = frame_coords[valid_mask].mean(axis=0)
            centered = frame_coords - ref_point
            
            # Scale
            scale = centered[valid_mask].std() + 1e-6
            coords[t] = centered / scale
    
    return normalized


def pad_or_truncate(sequence, target_length):
    """Pad or truncate sequence to fixed length"""
    T, J, D = sequence.shape
    
    if T == target_length:
        return sequence
    elif T > target_length:
        # Uniform sampling
        indices = np.linspace(0, T - 1, target_length, dtype=int)
        return sequence[indices]
    else:
        # Pad with zeros
        padding = np.zeros((target_length - T, J, D), dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)


class MediaPipeWorker:
    """Thread-safe MediaPipe worker"""
    
    def __init__(self, config):
        self.config = config
        self.holistic = None
        self.pose_indices = list(range(11, 25))
    
    def __enter__(self):
        # Initialize MediaPipe in the worker thread
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config['model_complexity'],
            smooth_landmarks=True,
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.holistic:
            self.holistic.close()
    
    def process_video(self, video_path, label):
        """Process one video"""
        try:
            # Extract frames
            frames = extract_video_frames_fast(video_path, self.config['max_frames'])
            if frames is None or len(frames) == 0:
                return None
            
            # Process frames
            sequences = []
            for frame in frames:
                results = self.holistic.process(frame)
                keypoints = extract_keypoints_from_frame(results, self.pose_indices)
                sequences.append(keypoints)
            
            # Stack into array
            sequence = np.stack(sequences, axis=0)  # (T, J, 3)
            
            # Normalize if enabled
            if self.config['normalize']:
                sequence = normalize_sequence_fast(sequence)
            
            # Pad/truncate to fixed length
            if self.config['max_frames']:
                sequence = pad_or_truncate(sequence, self.config['max_frames'])
            
            return {
                'video_path': str(video_path),
                'label': label,
                'sequence': sequence,
                'original_length': len(frames)
            }
            
        except Exception as e:
            # Silently fail, will be counted as failed
            return None


def process_video_wrapper(args):
    """Wrapper for processing a single video"""
    video_path, label, config = args
    with MediaPipeWorker(config) as worker:
        return worker.process_video(video_path, label)


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_split_optimized(split, config):
    """Process one split with threading or multiprocessing"""
    data_root = Path(config['data_root'])
    output_root = Path(config['output_root'])
    
    print(f"\n{'='*80}")
    print(f"Processing {split.upper()} split")
    print(f"{'='*80}")
    
    # Setup paths
    split_dir = data_root / split
    labels_file = data_root / f"{split}_labels.csv"
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    if not labels_file.exists():
        print(f"Warning: Labels file not found: {labels_file}")
        return
    
    df = pd.read_csv(labels_file)
    print(f"Found {len(df)} samples")
    
    # Prepare video paths and labels
    tasks = []
    for idx, row in df.iterrows():
        video_name = str(row[0])
        if not video_name.endswith('.mp4'):
            video_name = f"{video_name}_color.mp4"
        
        video_path = split_dir / video_name
        if video_path.exists():
            tasks.append((video_path, int(row[1]), config))
    
    print(f"Processing {len(tasks)} videos with {config['num_workers']} workers...")
    print(f"Mode: {'Threading' if config['use_threading'] else 'Multiprocessing'}")
    
    # Process with threading (safer for MediaPipe)
    metadata = []
    failed = []
    
    if config['use_threading']:
        # Use ThreadPoolExecutor (safer for MediaPipe)
        with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
            futures = [executor.submit(process_video_wrapper, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"{split}"):
                result = future.result()
                
                if result is not None:
                    video_path = Path(result['video_path'])
                    sequence = result['sequence']
                    label = result['label']
                    
                    # Save sequence
                    video_name = video_path.stem
                    output_path = output_dir / f"{video_name}.npy"
                    np.save(output_path, sequence)
                    
                    # Add to metadata
                    metadata.append({
                        'video_name': video_name,
                        'label': label,
                        'original_length': result['original_length'],
                        'num_keypoints': sequence.shape[1],
                        'file_path': str(output_path.relative_to(output_root))
                    })
                else:
                    failed.append("unknown")
                
                # Periodic garbage collection
                if len(metadata) % 100 == 0:
                    gc.collect()
    
    else:
        # Use ProcessPoolExecutor (may have issues on Windows)
        print("WARNING: Using multiprocessing mode. If you see crashes, set use_threading=True")
        
        # Process in chunks to avoid memory issues
        chunk_size = config.get('chunk_size', 100)
        
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i+chunk_size]
            
            with ProcessPoolExecutor(max_workers=config['num_workers']) as executor:
                results = list(tqdm(
                    executor.map(process_video_wrapper, chunk_tasks),
                    total=len(chunk_tasks),
                    desc=f"{split} chunk {i//chunk_size + 1}"
                ))
            
            # Save chunk results
            for result in results:
                if result is not None:
                    video_path = Path(result['video_path'])
                    sequence = result['sequence']
                    label = result['label']
                    
                    video_name = video_path.stem
                    output_path = output_dir / f"{video_name}.npy"
                    np.save(output_path, sequence)
                    
                    metadata.append({
                        'video_name': video_name,
                        'label': label,
                        'original_length': result['original_length'],
                        'num_keypoints': sequence.shape[1],
                        'file_path': str(output_path.relative_to(output_root))
                    })
                else:
                    failed.append("unknown")
            
            gc.collect()
    
    # Save metadata
    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_file = output_root / f"{split}_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        print(f"\nSaved metadata: {metadata_file}")
        print(f"Stats - Mean length: {metadata_df['original_length'].mean():.1f}, "
              f"Keypoints: {metadata_df['num_keypoints'].iloc[0]}")
    
    if failed:
        print(f"Failed: {len(failed)} videos")
    
    print(f"{split} complete: {len(metadata)} successful")


def main():
    """Main preprocessing pipeline"""
    config = CONFIG
    
    print("=" * 80)
    print("AUTSL Dataset Preprocessing - MediaPipe Safe Version")
    print("=" * 80)
    print(f"Data root: {config['data_root']}")
    print(f"Output root: {config['output_root']}")
    print(f"Workers: {config['num_workers']}")
    print(f"Max frames: {config['max_frames']}")
    print(f"Normalize: {config['normalize']}")
    print(f"Threading mode: {config['use_threading']}")
    print("=" * 80)
    
    # Validate paths
    data_root = Path(config['data_root'])
    if not data_root.exists():
        print(f"\nERROR: Data root not found: {data_root}")
        print("Please edit CONFIG['data_root'] at the top of this script")
        return
    
    # Create output directory
    output_root = Path(config['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in config['splits']:
        process_split_optimized(split, config)
    
    # Save config
    config_file = output_root / 'preprocess_config.json'
    config_save = {k: v for k, v in config.items() if k != 'use_threading'}
    with open(config_file, 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()