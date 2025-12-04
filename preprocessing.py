import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
import logging
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 14 upper body joints optimized for sign language (from 33 pose landmarks)
# Prioritizing shoulders, elbows, wrists which are critical for sign language
UPPER_BODY_IDX = np.array([
    0,      # nose (reference point)
    11, 12, # shoulders (left, right)
    13, 14, # elbows (left, right)
    15, 16, # wrists (left, right)
    23, 24, # hips (left, right) - body orientation
    7, 8,   # ears (left, right) - head orientation
    9, 10,  # mouth corners (left, right) - facial expressions
    1,      # added to make 14 total - left eye inner
], dtype=np.int32)

class SignLanguageAugmentation:
    """Advanced augmentation for sign language data preserving semantic meaning."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def spatial_scale(self, landmarks: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """Randomly scale the entire sign space (Eq. 8 in paper)."""
        scale = self.rng.uniform(*scale_range)
        # Scale x, y coordinates around per-frame center
        landmarks_aug = landmarks.copy()
        for t in range(len(landmarks)):
            center = landmarks[t, :, :2].mean(axis=0)
            landmarks_aug[t, :, :2] = (landmarks[t, :, :2] - center) * scale + center
        return landmarks_aug
    
    def spatial_translate(self, landmarks: np.ndarray, translate_range: float = 0.05) -> np.ndarray:
        """Translate all landmarks by a random offset."""
        tx = self.rng.uniform(-translate_range, translate_range)
        ty = self.rng.uniform(-translate_range, translate_range)
        landmarks_aug = landmarks.copy()
        landmarks_aug[:, :, 0] += tx
        landmarks_aug[:, :, 1] += ty
        return landmarks_aug
    
    def spatial_rotate(self, landmarks: np.ndarray, angle_range: float = 5) -> np.ndarray:
        """Rotate landmarks around per-frame center in 2D plane (Eq. 5 in paper: ±5°)."""
        angle = self.rng.uniform(-angle_range, angle_range)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        landmarks_aug = landmarks.copy()
        
        # Apply rotation per-frame
        for t in range(len(landmarks)):
            center = landmarks[t, :, :2].mean(axis=0)
            centered = landmarks[t, :, :2] - center
            rotated = np.stack([
                centered[:, 0] * cos_a - centered[:, 1] * sin_a,
                centered[:, 0] * sin_a + centered[:, 1] * cos_a
            ], axis=-1)
            landmarks_aug[t, :, :2] = rotated + center
        return landmarks_aug
    
    def temporal_scale(self, landmarks: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Time stretch/compress by resampling."""
        scale = self.rng.uniform(*scale_range)
        new_length = max(int(len(landmarks) * scale), 1)
        indices = np.linspace(0, len(landmarks) - 1, new_length)
        
        # Vectorized interpolation
        result = np.zeros((new_length, landmarks.shape[1], landmarks.shape[2]), dtype=landmarks.dtype)
        for joint in range(landmarks.shape[1]):
            for coord in range(landmarks.shape[2]):
                result[:, joint, coord] = np.interp(indices, np.arange(len(landmarks)), landmarks[:, joint, coord])
        return result
    
    def temporal_masking(self, landmarks: np.ndarray, mask_ratio: float = 0.15) -> np.ndarray:
        """Temporal masking: zero out random frames (Eq. 7 in paper)."""
        landmarks_aug = landmarks.copy()
        num_frames = len(landmarks)
        num_mask = int(num_frames * mask_ratio)
        
        if num_mask > 0:
            mask_indices = self.rng.choice(num_frames, num_mask, replace=False)
            landmarks_aug[mask_indices] = 0.0
        
        return landmarks_aug
    
    def gaussian_noise(self, landmarks: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """Add small Gaussian noise to coordinates (Eq. 6 in paper: σ=0.01)."""
        noise = self.rng.normal(0, noise_std, landmarks.shape).astype(landmarks.dtype)
        return landmarks + noise
    
    def horizontal_flip(self, landmarks: np.ndarray) -> np.ndarray:
        """Mirror transformation with left-right joint swap (Eq. 9 in paper)."""
        landmarks_aug = landmarks.copy()
        
        # FIXED: Flip x-coordinates correctly for normalized coordinates [0, 1]
        landmarks_aug[:, :, 0] = 1.0 - landmarks_aug[:, :, 0]
        
        # FIXED: Swap left-right joints using correct indices (0-based positions in 14-element array)
        # swap_pairs uses positions in the compacted UPPER_BODY_IDX array
        swap_pairs = [
            (1, 2),   # shoulders (indices 11, 12 in MediaPipe -> positions 1, 2 in our array)
            (3, 4),   # elbows (13, 14 -> 3, 4)
            (5, 6),   # wrists (15, 16 -> 5, 6)
            (7, 8),   # hips (23, 24 -> 7, 8)
            (9, 10),  # ears (7, 8 -> 9, 10)
            (11, 12)  # mouth corners (9, 10 -> 11, 12)
        ]
        
        for left_idx, right_idx in swap_pairs:
            landmarks_aug[:, [left_idx, right_idx]] = landmarks_aug[:, [right_idx, left_idx]].copy()
        
        # FIXED: Swap hands with proper shape validation
        # Validate shape before swapping
        if landmarks_aug.shape[1] == 56:  # 14 pose + 21 left hand + 21 right hand
            landmarks_aug[:, 14:35], landmarks_aug[:, 35:56] = \
                landmarks_aug[:, 35:56].copy(), landmarks_aug[:, 14:35].copy()
        
        return landmarks_aug
    
    def augment(self, landmarks: np.ndarray, augmentation_config: Dict[str, float]) -> np.ndarray:
        """Apply multiple augmentations based on paper's specified probabilities."""
        aug_landmarks = landmarks.copy()
        
        # Spatial rotation (50% in paper)
        if self.rng.rand() < augmentation_config.get('spatial_rotate_prob', 0.5):
            aug_landmarks = self.spatial_rotate(aug_landmarks, angle_range=5)
        
        # Gaussian noise (30% in paper)
        if self.rng.rand() < augmentation_config.get('gaussian_noise_prob', 0.3):
            aug_landmarks = self.gaussian_noise(aug_landmarks, noise_std=0.01)
        
        # Temporal masking (10% in paper)
        if self.rng.rand() < augmentation_config.get('temporal_mask_prob', 0.1):
            aug_landmarks = self.temporal_masking(aug_landmarks, mask_ratio=0.15)
        
        # Spatial scaling (30% in paper)
        if self.rng.rand() < augmentation_config.get('spatial_scale_prob', 0.3):
            aug_landmarks = self.spatial_scale(aug_landmarks, scale_range=(0.95, 1.05))
        
        # Horizontal flip (20% in paper)
        if self.rng.rand() < augmentation_config.get('horizontal_flip_prob', 0.2):
            aug_landmarks = self.horizontal_flip(aug_landmarks)
        
        return aug_landmarks


class LandmarkExtractor:
    """Optimized landmark extraction with normalization per paper."""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.3,
                 min_tracking_confidence: float = 0.3,
                 enable_segmentation: bool = False):
        """Initialize MediaPipe Holistic model."""
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.holistic.close()
    
    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 56 landmarks from a single frame.
        
        Returns:
            Array of shape (56, 3) containing [x, y, z] coordinates
        """
        # Convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        
        res = self.holistic.process(img_rgb)
        
        # Pre-allocate output
        landmarks = np.zeros((56, 3), dtype=np.float32)
        
        # Extract pose landmarks (14 upper body)
        if res.pose_landmarks:
            pose_lms = res.pose_landmarks.landmark
            for i, idx in enumerate(UPPER_BODY_IDX):
                lm = pose_lms[idx]
                landmarks[i] = [lm.x, lm.y, lm.z]
        
        # Extract left hand (21 points)
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                landmarks[14 + i] = [lm.x, lm.y, lm.z]
        
        # Extract right hand (21 points)
        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                landmarks[35 + i] = [lm.x, lm.y, lm.z]
        
        return landmarks
    
    @staticmethod
    def normalize_sequence(sequence: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Apply per-frame normalization as per equations (2-4) in paper.
        
        For each frame t:
            μ_t = mean of all joints (Eq. 3)
            σ_t = std deviation (Eq. 4)
            x'_t,j = (x_t,j - μ_t) / σ_t (Eq. 2)
        
        Args:
            sequence: (T, J, 3) array
            epsilon: Small constant for numerical stability
            
        Returns:
            Normalized sequence of same shape
        """
        normalized = np.zeros_like(sequence)
        
        for t in range(len(sequence)):
            frame = sequence[t]  # (J, 3)
            
            # Compute mean across all joints (Eq. 3)
            mu_t = frame.mean(axis=0)  # (3,)
            
            # Compute std deviation (Eq. 4)
            centered = frame - mu_t
            sigma_t = np.sqrt((centered ** 2).mean(axis=0)) + epsilon  # (3,)
            
            # Normalize (Eq. 2)
            normalized[t] = centered / sigma_t
        
        return normalized
    
    def extract_from_video(self, 
                          video_path: str,
                          fixed_frames: int = 64,
                          skip_frames: int = 0) -> Tuple[np.ndarray, dict]:
        """Extract landmarks from video (WITHOUT normalization - done later)."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get metadata
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0 or np.isnan(original_fps):
            original_fps = 30.0  # fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_list = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % (skip_frames + 1) == 0:
                    landmarks = self.extract_from_frame(frame)
                    landmarks_list.append(landmarks)
                
                frame_count += 1
                
        finally:
            cap.release()
        
        if not landmarks_list:
            return np.zeros((0, 56, 3), dtype=np.float32), {
                'total_frames': 0, 'processed_frames': 0,
                'original_fps': 0.0, 'shape': (0, 56, 3)
            }
        
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        
        # Resample to fixed frames (64 as per paper)
        if len(landmarks_array) != fixed_frames:
            landmarks_array = self._resample_sequence(landmarks_array, fixed_frames)
        
        # REMOVED: normalization here to avoid double normalization
        # Normalization will be done once in process_single_video after all resampling
        
        metadata = {
            'total_frames': frame_count,
            'processed_frames': len(landmarks_array),
            'original_fps': original_fps,
            'shape': landmarks_array.shape,
            'normalized': False  # Changed to False
        }
        
        return landmarks_array, metadata
    
    @staticmethod
    def _resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Resample using linear interpolation for smooth transitions."""
        if len(sequence) == 0:
            return np.zeros((target_length, 56, 3), dtype=np.float32)
        
        if len(sequence) == target_length:
            return sequence
        
        # Use linear interpolation for each joint and coordinate
        old_indices = np.arange(len(sequence))
        new_indices = np.linspace(0, len(sequence) - 1, target_length)
        
        resampled = np.zeros((target_length, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
        for joint in range(sequence.shape[1]):
            for coord in range(sequence.shape[2]):
                resampled[:, joint, coord] = np.interp(new_indices, old_indices, sequence[:, joint, coord])
        
        return resampled


# Global extractor for multiprocessing
extractor = None
augmentor = None

def init_worker(config):
    """Initialize extractor and augmentor once per worker."""
    global extractor, augmentor
    extractor = LandmarkExtractor(
        model_complexity=config['model_complexity'],
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    augmentor = SignLanguageAugmentation(seed=None)


def process_single_video(args):
    """Process single video with optional augmentation."""
    video_info, config = args
    
    try:
        video_base = video_info['video_base']
        video_path = video_info['video_path']
        label = video_info['label']
        sign_name = video_info['sign_name']
        output_path = video_info['output_path']
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract landmarks WITHOUT normalization
        landmarks, metadata = extractor.extract_from_video(
            str(video_path),
            fixed_frames=config['fixed_frames']
        )
        
        if landmarks.size == 0:
            raise RuntimeError("No frames extracted")
        
        # Ensure correct shape and resample if needed
        if len(landmarks) != config['fixed_frames']:
            landmarks = LandmarkExtractor._resample_sequence(landmarks, config['fixed_frames'])
        
        # FIXED: Normalize ONCE after resampling
        landmarks = LandmarkExtractor.normalize_sequence(landmarks)
        
        results = []
        
        # Save original
        original_path = output_path
        if config['use_float16']:
            landmarks_save = landmarks.astype(np.float16)
        else:
            landmarks_save = landmarks
        
        np.savez_compressed(original_path, data=landmarks_save)
        file_size = original_path.stat().st_size
        
        results.append({
            'video_name': video_base,
            'label': label,
            'sign_name': sign_name,
            'npz_path': str(original_path.relative_to(config['output_dir'])),
            'num_frames': config['fixed_frames'],
            'original_fps': metadata['original_fps'],
            'shape': str(landmarks.shape),
            'dtype': str(landmarks_save.dtype),
            'file_size_kb': file_size / 1024,
            'augmented': False,
            'normalized': True
        })
        
        # Generate augmentations if enabled
        if config['enable_augmentation'] and config['num_augmentations'] > 0:
            for aug_idx in range(config['num_augmentations']):
                # Apply augmentation
                aug_landmarks = augmentor.augment(landmarks, config['augmentation_config'])
                
                # Resample if needed
                if len(aug_landmarks) != config['fixed_frames']:
                    aug_landmarks = LandmarkExtractor._resample_sequence(aug_landmarks, config['fixed_frames'])
                    # Re-normalize after resampling
                    aug_landmarks = LandmarkExtractor.normalize_sequence(aug_landmarks)
                
                # Save augmented version
                aug_output_path = output_path.parent / f"{video_base}_aug{aug_idx}.npz"
                
                if config['use_float16']:
                    aug_landmarks_save = aug_landmarks.astype(np.float16)
                else:
                    aug_landmarks_save = aug_landmarks
                
                np.savez_compressed(aug_output_path, data=aug_landmarks_save)
                aug_file_size = aug_output_path.stat().st_size
                
                results.append({
                    'video_name': f"{video_base}_aug{aug_idx}",
                    'label': label,
                    'sign_name': sign_name,
                    'npz_path': str(aug_output_path.relative_to(config['output_dir'])),
                    'num_frames': config['fixed_frames'],
                    'original_fps': metadata['original_fps'],
                    'shape': str(aug_landmarks.shape),
                    'dtype': str(aug_landmarks_save.dtype),
                    'file_size_kb': aug_file_size / 1024,
                    'augmented': True,
                    'augmentation_index': aug_idx,
                    'normalized': True
                })
        
        return (True, results)
        
    except Exception as e:
        return (False, [{
            'video_name': video_info.get('video_base', 'unknown'),
            'error': str(e)
        }])


def process_autsl_dataset(
    dataset_path: str,
    output_dir: str = "processed_autsl",
    splits: List[str] = ['train', 'val', 'test'],
    fixed_frames: int = 64,
    model_complexity: int = 0,
    use_float16: bool = True,
    num_workers: Optional[int] = None,
    enable_augmentation: bool = True,
    num_augmentations: int = 3,
    augmentation_config: Optional[Dict] = None
):
    """
    Process AUTSL dataset following CorrFormer-Lite paper specifications.
    
    Args:
        dataset_path: Root path to AUTSL dataset
        output_dir: Output directory
        splits: Dataset splits to process
        fixed_frames: Frames per sequence (64 as per paper)
        model_complexity: 0=fastest, 1=balanced, 2=accurate
        use_float16: Use float16 (50% size reduction)
        num_workers: Parallel processes (default: CPU count - 1)
        enable_augmentation: Enable data augmentation
        num_augmentations: Number of augmented versions per video
        augmentation_config: Custom augmentation probabilities
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Default augmentation config matching paper
    if augmentation_config is None:
        augmentation_config = {
            'spatial_rotate_prob': 0.5,    # 50% as per paper
            'gaussian_noise_prob': 0.3,    # 30% as per paper
            'temporal_mask_prob': 0.1,     # 10% as per paper
            'spatial_scale_prob': 0.3,     # 30% as per paper
            'horizontal_flip_prob': 0.2    # 20% as per paper
        }
    
    print(f"🚀 CorrFormer-Lite Preprocessing Pipeline")
    print(f"{'='*60}")
    print(f"📊 Configuration:")
    print(f"   Workers: {num_workers}")
    print(f"   Model complexity: {model_complexity}")
    print(f"   Fixed frames: {fixed_frames}")
    print(f"   Normalization: Enabled (per-frame, Eq. 2-4)")
    print(f"   Augmentation: {'Enabled' if enable_augmentation else 'Disabled'}")
    if enable_augmentation:
        print(f"   Augmentations per video: {num_augmentations}")
        print(f"   Aug config: {augmentation_config}")
    
    # Create directories
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load class mapping
    class_mapping_path = dataset_path / "SignList_ClassId_TR_EN.csv"
    class_mapping = None
    if class_mapping_path.exists():
        class_mapping = pd.read_csv(class_mapping_path)
        class_mapping.to_csv(output_dir / "class_mapping.csv", index=False)
        print(f"✓ Loaded class mapping: {len(class_mapping)} classes")
    
    processing_stats = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        labels_path = dataset_path / f"{split}_labels.csv"
        if not labels_path.exists():
            print(f"⚠ {labels_path} not found, skipping...")
            continue
        
        labels_df = pd.read_csv(labels_path)
        
        # Auto-detect columns
        if 'signer' in str(labels_df.columns[0]).lower():
            labels_df = pd.read_csv(labels_path, header=None, names=['video', 'label'])
        
        video_col = next((c for c in ['video', 'sample', 'name'] if c in labels_df.columns), labels_df.columns[0])
        label_col = next((c for c in ['label', 'class', 'ClassId'] if c in labels_df.columns), labels_df.columns[1])
        
        print(f"📁 Loaded {len(labels_df)} videos")
        print(f"📋 Using columns: video='{video_col}', label='{label_col}'")
        
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        
        # Prepare work items
        work_items = []
        for idx, row in labels_df.iterrows():
            video_base = str(row[video_col]).replace('_color', '')
            video_path = split_dir / f"{video_base}_color.mp4"
            if not video_path.exists():
                video_path = split_dir / f"{video_base}.mp4"
            if not video_path.exists():
                continue
            
            # FIXED: Ensure label type consistency
            label = int(row[label_col]) if pd.notna(row[label_col]) else 0
            sign_name = "unknown"
            
            if class_mapping is not None:
                # FIXED: Ensure type matching for comparison
                class_row = class_mapping[class_mapping['ClassId'].astype(int) == label]
                if not class_row.empty:
                    sign_name = class_row.iloc[0].get('EN', 'unknown')
            
            output_path = output_dir / split / f"{video_base}.npz"
            work_items.append({
                'video_base': video_base,
                'video_path': video_path,
                'label': label,
                'sign_name': sign_name,
                'output_path': output_path
            })
        
        # Config for workers
        config = {
            'fixed_frames': fixed_frames,
            'model_complexity': model_complexity,
            'use_float16': use_float16,
            'output_dir': output_dir,
            'enable_augmentation': enable_augmentation and split == 'train',
            'num_augmentations': num_augmentations,
            'augmentation_config': augmentation_config
        }
        
        # Process with multiprocessing
        metadata_records = []
        failed_videos = []
        total_size_bytes = 0
        
        print(f"⚡ Processing {len(work_items)} videos...")
        
        with Pool(processes=num_workers, initializer=init_worker, initargs=(config,)) as pool:
            results = pool.imap(
                process_single_video,
                [(item, config) for item in work_items],
                chunksize=1
            )
            
            for success, result_list in tqdm(results, total=len(work_items), desc=f"{split}"):
                if success:
                    for result in result_list:
                        metadata_records.append(result)
                        total_size_bytes += result['file_size_kb'] * 1024
                else:
                    failed_videos.extend(result_list)
        
        # Save metadata
        if metadata_records:
            metadata_df = pd.DataFrame(metadata_records)
            csv_path = output_dir / f"{split}_metadata.csv"
            metadata_df.to_csv(csv_path, index=False)
            
            orig_count = len(metadata_df[~metadata_df['augmented']])
            aug_count = len(metadata_df[metadata_df['augmented']])
            
            print(f"\n✅ Saved metadata to {csv_path}")
            print(f"  📹 Original videos: {orig_count}")
            if aug_count > 0:
                print(f"  🎭 Augmented samples: {aug_count}")
            print(f"  📊 Total samples: {len(metadata_df)}")
            print(f"  💾 Total size: {total_size_bytes / (1024**2):.2f} MB")
        
        if failed_videos:
            print(f"  ⚠ Failed: {len(failed_videos)} videos")
        
        processing_stats[split] = {
            'total_videos': len(work_items),
            'successful': len(metadata_records),
            'failed': len(failed_videos),
            'total_size_mb': total_size_bytes / (1024**2)
        }
    
    # Save processing info
    info_path = output_dir / "processing_info.json"
    with open(info_path, 'w') as f:
        json.dump({
            'dataset_path': str(dataset_path),
            'fixed_frames': fixed_frames,
            'model_complexity': model_complexity,
            'use_float16': use_float16,
            'num_workers': num_workers,
            'enable_augmentation': enable_augmentation,
            'num_augmentations': num_augmentations,
            'augmentation_config': augmentation_config,
            'normalization': 'per-frame (Eq. 2-4)',
            'processing_stats': processing_stats
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Processing complete!")
    print(f"{'='*60}")
    
    total_size = sum(s['total_size_mb'] for s in processing_stats.values())
    total_samples = sum(s['successful'] for s in processing_stats.values())
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  ✅ Total samples: {total_samples}")
    print(f"  💾 Storage: {total_size:.2f} MB")
    print(f"  📈 Avg per sample: {(total_size / total_samples):.3f} MB")


if __name__ == "__main__":
    process_autsl_dataset(
        dataset_path="autsl/",
        output_dir="processed_autsl",
        splits=['train', 'val', 'test'],
        fixed_frames=64,
        model_complexity=0,
        use_float16=True,
        num_workers=None,
        enable_augmentation=True,
        num_augmentations=3,
        augmentation_config=None  # Uses paper's specified probabilities
    )