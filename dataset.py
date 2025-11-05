"""
dataset.py - PyTorch Dataset for AUTSL Pose-based Sign Language Recognition

Supports:
- Train/Val/Test splits
- Data augmentation (rotation, noise, temporal masking)
- Semantic joint embeddings preparation
- Fixed-length sequences (64 frames, 56 joints, 3 coords)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class PoseSignDataset(Dataset):
    """
    AUTSL Pose Dataset for Sign Language Recognition
    
    Args:
        data_root: Root directory containing processed data
        split: 'train', 'val', or 'test'
        max_frames: Maximum sequence length (default: 64)
        num_keypoints: Number of joints per frame (default: 56)
        augment: Whether to apply data augmentation (default: False)
        aug_config: Augmentation configuration dict
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_frames: int = 64,
        num_keypoints: int = 56,
        augment: bool = False,
        aug_config: Optional[dict] = None
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.max_frames = max_frames
        self.num_keypoints = num_keypoints
        self.augment = augment and (split == 'train')  # Only augment training
        
        # Default augmentation config
        self.aug_config = aug_config or {
            'rotation_range': 5.0,       # degrees
            'noise_std': 0.01,           # Gaussian noise
            'temporal_mask_prob': 0.1,   # Probability of temporal masking
            'temporal_mask_ratio': 0.15, # Ratio of frames to mask
            'spatial_scale': 0.05,       # Random spatial scaling
        }
        
        # Load metadata
        metadata_file = self.data_root / f"{split}_metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")
        
        self.metadata = pd.read_csv(metadata_file)
        
        # Joint type semantic groups (for semantic embeddings)
        self.joint_groups = self._define_joint_groups()
        
        print(f"Loaded {split} split: {len(self.metadata)} samples")
        print(f"Augmentation: {'ON' if self.augment else 'OFF'}")
        
    def _define_joint_groups(self) -> dict:
        """
        Define semantic groupings for joints
        56 joints = 14 pose (upper body) + 21 left hand + 21 right hand
        """
        groups = {
            'pose': list(range(0, 14)),           # Upper body joints
            'left_hand': list(range(14, 35)),     # Left hand joints
            'right_hand': list(range(35, 56)),    # Right hand joints
        }
        
        # Create joint type IDs for semantic embeddings
        joint_types = np.zeros(self.num_keypoints, dtype=np.int64)
        joint_types[groups['pose']] = 0
        joint_types[groups['left_hand']] = 1
        joint_types[groups['right_hand']] = 2
        
        return {
            'groups': groups,
            'types': joint_types  # (56,) array with type IDs
        }
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence: (T, J, 3) pose sequence
            joint_types: (J,) semantic joint type IDs
            label: (,) class label
        """
        # Load sample
        row = self.metadata.iloc[idx]
        file_path = self.data_root / row['file_path']
        
        # Load pose sequence
        sequence = np.load(file_path).astype(np.float32)  # (T, J, 3)
        label = int(row['label'])
        
        # Apply augmentation
        if self.augment:
            sequence = self._augment(sequence)
        
        # Convert to tensors
        sequence = torch.from_numpy(sequence)  # (T, J, 3)
        joint_types = torch.from_numpy(self.joint_groups['types'])  # (J,)
        label = torch.tensor(label, dtype=torch.long)
        
        return sequence, joint_types, label
    
    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to pose sequence
        
        Args:
            sequence: (T, J, 3) array
        Returns:
            augmented: (T, J, 3) array
        """
        T, J, C = sequence.shape
        aug_seq = sequence.copy()
        
        # 1. Spatial rotation (around Z-axis, in XY plane)
        if random.random() < 0.5:
            angle = np.random.uniform(
                -self.aug_config['rotation_range'],
                self.aug_config['rotation_range']
            )
            aug_seq = self._rotate_sequence(aug_seq, angle)
        
        # 2. Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(
                0, 
                self.aug_config['noise_std'], 
                size=aug_seq.shape
            ).astype(np.float32)
            aug_seq += noise
        
        # 3. Temporal masking (set random frames to zero)
        if random.random() < self.aug_config['temporal_mask_prob']:
            mask_length = int(T * self.aug_config['temporal_mask_ratio'])
            mask_start = random.randint(0, max(1, T - mask_length))
            aug_seq[mask_start:mask_start + mask_length] = 0.0
        
        # 4. Spatial scaling
        if random.random() < 0.3:
            scale = 1.0 + np.random.uniform(
                -self.aug_config['spatial_scale'],
                self.aug_config['spatial_scale']
            )
            aug_seq[:, :, :2] *= scale  # Only scale XY, not confidence
        
        # 5. Horizontal flip (swap left/right)
        if random.random() < 0.2:
            aug_seq = self._flip_sequence(aug_seq)
        
        return aug_seq
    
    def _rotate_sequence(self, sequence: np.ndarray, angle: float) -> np.ndarray:
        """Rotate sequence in XY plane"""
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)
        
        # Apply to XY coordinates
        T, J, C = sequence.shape
        rotated = sequence.copy()
        xy = sequence[:, :, :2]  # (T, J, 2)
        
        # Reshape for matrix multiplication
        xy_flat = xy.reshape(-1, 2)  # (T*J, 2)
        xy_rotated = xy_flat @ rot_matrix.T
        rotated[:, :, :2] = xy_rotated.reshape(T, J, 2)
        
        return rotated
    
    def _flip_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Flip sequence horizontally (swap left/right hands)"""
        flipped = sequence.copy()
        
        # Flip X coordinates
        flipped[:, :, 0] = -flipped[:, :, 0]
        
        # Swap left and right hand joints
        left_hand = self.joint_groups['groups']['left_hand']
        right_hand = self.joint_groups['groups']['right_hand']
        
        temp = flipped[:, left_hand, :].copy()
        flipped[:, left_hand, :] = flipped[:, right_hand, :]
        flipped[:, right_hand, :] = temp
        
        return flipped
    
    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a specific sample"""
        row = self.metadata.iloc[idx]
        return {
            'video_name': row['video_name'],
            'label': int(row['label']),
            'original_length': int(row['original_length']),
            'file_path': str(row['file_path'])
        }


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader
    
    Args:
        batch: List of (sequence, joint_types, label) tuples
    
    Returns:
        sequences: (B, T, J, 3)
        joint_types: (B, J)
        labels: (B,)
    """
    sequences, joint_types, labels = zip(*batch)
    
    # Stack into batch
    sequences = torch.stack(sequences, dim=0)      # (B, T, J, 3)
    joint_types = torch.stack(joint_types, dim=0)  # (B, J)
    labels = torch.stack(labels, dim=0)            # (B,)
    
    return sequences, joint_types, labels


def create_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = False,
    aug_config: Optional[dict] = None,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for a specific split
    
    Args:
        data_root: Root directory of processed data
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply augmentation
        aug_config: Augmentation configuration
        shuffle: Whether to shuffle (default: True for train, False otherwise)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = PoseSignDataset(
        data_root=data_root,
        split=split,
        augment=augment,
        aug_config=aug_config
    )
    
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=(split == 'train'),  # Drop last incomplete batch for training
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader


def get_dataset_statistics(data_root: str) -> dict:
    """
    Get statistics about the dataset
    
    Args:
        data_root: Root directory of processed data
    
    Returns:
        Dictionary with dataset statistics
    """
    data_root = Path(data_root)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        metadata_file = data_root / f"{split}_metadata.csv"
        if not metadata_file.exists():
            continue
        
        df = pd.read_csv(metadata_file)
        stats[split] = {
            'num_samples': len(df),
            'num_classes': df['label'].nunique(),
            'class_distribution': df['label'].value_counts().to_dict(),
            'mean_length': df['original_length'].mean(),
            'std_length': df['original_length'].std(),
        }
    
    return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    """Test the dataset and dataloader"""
    
    # Configuration
    DATA_ROOT = 'data/processed_fast'
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Augmentation config
    aug_config = {
        'rotation_range': 5.0,
        'noise_std': 0.01,
        'temporal_mask_prob': 0.1,
        'temporal_mask_ratio': 0.15,
        'spatial_scale': 0.05,
    }
    
    print("="*80)
    print("Testing AUTSL Dataset and DataLoader")
    print("="*80)
    
    # Get dataset statistics
    print("\n📊 Dataset Statistics:")
    stats = get_dataset_statistics(DATA_ROOT)
    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Samples: {split_stats['num_samples']}")
        print(f"  Classes: {split_stats['num_classes']}")
        print(f"  Avg length: {split_stats['mean_length']:.1f} ± {split_stats['std_length']:.1f}")
    
    # Create dataloaders
    print("\n🔄 Creating DataLoaders...")
    train_loader = create_dataloader(
        DATA_ROOT, 'train', 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        augment=True,
        aug_config=aug_config
    )
    
    val_loader = create_dataloader(
        DATA_ROOT, 'val',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        augment=False
    )
    
    test_loader = create_dataloader(
        DATA_ROOT, 'test',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        augment=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    for sequences, joint_types, labels in train_loader:
        print(f"\n  Batch shapes:")
        print(f"    Sequences: {sequences.shape}")      # (B, T, J, 3)
        print(f"    Joint types: {joint_types.shape}")  # (B, J)
        print(f"    Labels: {labels.shape}")            # (B,)
        
        print(f"\n  Data ranges:")
        print(f"    Sequence min/max: {sequences.min():.3f} / {sequences.max():.3f}")
        print(f"    Joint types unique: {torch.unique(joint_types)}")
        print(f"    Labels min/max: {labels.min()} / {labels.max()}")
        
        # Check for NaN
        if torch.isnan(sequences).any():
            print("  ⚠️  WARNING: NaN values detected!")
        else:
            print("  ✅ No NaN values")
        
        break 
    
    print("\n✅ Dataset and DataLoader working correctly!")
    print("="*80)