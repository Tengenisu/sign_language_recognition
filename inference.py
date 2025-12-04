"""
inference.py - Single Video Inference for Sign Language Translation

Usage:
    python inference.py --video path/to/video.mp4 --checkpoint checkpoints/exp_name/best_model.pth
    
    Or with auto-detection:
    python inference.py --video path/to/video.mp4
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

import cv2
import mediapipe as mp

# Import your custom modules
from model import create_model

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'checkpoint_path': None,  # Will auto-detect if None
    'data_root': 'processed',  # Where normalization_stats.json is
    'device': 'cuda',  # 'cuda' or 'cpu'
    'top_k': 5,  # Show top-k predictions
}

DEFAULT_PREPROCESS_SETTINGS = {
    'max_frames': 64,
    'normalize': True,
    'model_complexity': 0,
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.3,
}

POSE_LANDMARKS = list(range(11, 25))  # Upper body landmarks used during preprocessing
NUM_HAND_LANDMARKS = 21  # MediaPipe hand landmarks per side
NUM_JOINTS = len(POSE_LANDMARKS) + 2 * NUM_HAND_LANDMARKS

JOINT_TYPE_TEMPLATE = torch.tensor(
    [0] * len(POSE_LANDMARKS) +
    [1] * NUM_HAND_LANDMARKS +
    [2] * NUM_HAND_LANDMARKS,
    dtype=torch.long
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_preprocess_settings(data_root: str) -> Dict[str, float]:
    """Load preprocessing settings used during dataset creation."""
    settings = DEFAULT_PREPROCESS_SETTINGS.copy()
    config_path = Path(data_root) / 'preprocess_config.json'

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)

            for key in ['max_frames', 'normalize', 'model_complexity',
                        'min_detection_confidence', 'min_tracking_confidence']:
                if key in file_config:
                    settings[key] = file_config[key]
        except Exception as exc:
            print(f"⚠️  Failed to read preprocess_config.json: {exc}. Using defaults.")

    return settings


def get_joint_type_tensor(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """Return semantic joint type IDs matching the training pipeline."""
    if num_joints != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joints, received {num_joints}")
    return JOINT_TYPE_TEMPLATE.clone()


def sample_video_frames(video_path: Path, max_frames: Optional[int]) -> list:
    """Uniformly sample frames from the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    if max_frames and total_frames > max_frames:
        target_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        target_indices = np.arange(total_frames, dtype=int)

    target_indices = np.unique(target_indices)
    frames = []
    next_idx = 0

    for frame_idx in range(total_frames):
        if next_idx >= len(target_indices):
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_indices[next_idx]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            next_idx += 1

    cap.release()

    if not frames:
        raise ValueError(f"Failed to sample frames from video: {video_path}")

    return frames


def extract_keypoints_from_results(results) -> np.ndarray:
    """Convert MediaPipe holistic results into a (J, 3) array."""
    keypoints = []

    if results.pose_landmarks:
        for idx in POSE_LANDMARKS:
            lm = results.pose_landmarks.landmark[idx]
            keypoints.append([lm.x, lm.y, lm.visibility])
    else:
        keypoints.extend([[0.0, 0.0, 0.0]] * len(POSE_LANDMARKS))

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([[0.0, 0.0, 0.0]] * NUM_HAND_LANDMARKS)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([[0.0, 0.0, 0.0]] * NUM_HAND_LANDMARKS)

    return np.asarray(keypoints, dtype=np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Apply per-frame normalization used during preprocessing."""
    normalized = sequence.copy()
    coords = normalized[:, :, :2]
    confidence = normalized[:, :, 2:3]

    for t in range(normalized.shape[0]):
        frame_conf = confidence[t].squeeze(-1)
        valid_mask = frame_conf > 0.1
        if valid_mask.any():
            ref_point = coords[t, valid_mask].mean(axis=0)
            coords[t] = coords[t] - ref_point
            scale = coords[t, valid_mask].std() + 1e-6
            coords[t] = coords[t] / scale

    normalized[:, :, :2] = coords
    return normalized


def pad_or_truncate(sequence: np.ndarray, target_length: Optional[int]) -> np.ndarray:
    """Pad or uniformly sample sequence to the target length."""
    if target_length is None:
        return sequence

    T, J, C = sequence.shape
    if T == target_length:
        return sequence
    if T > target_length:
        indices = np.linspace(0, T - 1, target_length, dtype=int)
        return sequence[indices]

    padding = np.zeros((target_length - T, J, C), dtype=sequence.dtype)
    return np.concatenate([sequence, padding], axis=0)


def extract_keypoints_from_video(video_path: Path, settings: Dict[str, float]) -> tuple[np.ndarray, int]:
    """Extract normalized keypoint sequence from a video."""
    frames = sample_video_frames(video_path, settings.get('max_frames'))

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(settings.get('model_complexity', 0)),
        smooth_landmarks=True,
        min_detection_confidence=float(settings.get('min_detection_confidence', 0.3)),
        min_tracking_confidence=float(settings.get('min_tracking_confidence', 0.3))
    )

    sequences = []
    try:
        for frame in frames:
            results = holistic.process(frame)
            keypoints = extract_keypoints_from_results(results)
            sequences.append(keypoints)
    finally:
        holistic.close()

    if not sequences:
        raise ValueError("MediaPipe failed to produce any keypoints")

    sequence = np.stack(sequences, axis=0).astype(np.float32)

    if settings.get('normalize', True):
        sequence = normalize_sequence(sequence)

    sequence = pad_or_truncate(sequence, settings.get('max_frames'))

    return sequence, len(frames)


def normalize_keypoints(keypoints: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    """Apply dataset-level normalization if stats are provided."""
    if mean is None or std is None:
        return keypoints

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.ndim == 1:
        if mean.size == keypoints.shape[-1]:
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
        else:
            mean = mean.reshape(1, keypoints.shape[1], keypoints.shape[2])
            std = std.reshape(1, keypoints.shape[1], keypoints.shape[2])
    elif mean.ndim == 2:
        mean = mean.reshape(1, *mean.shape)
        std = std.reshape(1, *std.shape)
    elif mean.ndim != 3:
        raise ValueError(f"Unsupported normalization statistics shape: {mean.shape}")

    return (keypoints - mean) / (std + 1e-6)


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model checkpoint and configuration"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    # Load config
    config_path = checkpoint_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration")
    else:
        # Default config
        config = {
            'num_classes': 226,
            'model_size': 'base',
            'dropout': 0.1
        }
        print(f"⚠ Config not found, using defaults")
    
    # Create model
    print(f"🏗️  Creating model (size={config['model_size']}, classes={config['num_classes']})")
    model = create_model(
        num_classes=config.get('num_classes', 226),
        model_size=config.get('model_size', 'base'),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Get training info
    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('best_acc', checkpoint.get('best_val_acc', checkpoint.get('val_acc', 'unknown')))
    print(f"   Epoch: {epoch}, Val Accuracy: {val_acc}")
    
    return model, config


def load_label_mapping(data_root='processed'):
    """Load label mapping (index -> class name)"""
    mapping_path = Path(data_root) / 'label_mapping.json'
    
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            label_mapping = json.load(f)
        print(f"✓ Loaded label mapping: {len(label_mapping)} classes")
        return label_mapping
    else:
        print(f"⚠ Label mapping not found, using indices")
        return None


def load_normalization_stats(data_root='processed'):
    """Load normalization statistics"""
    stats_path = Path(data_root) / 'normalization_stats.json'
    
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"✓ Loaded normalization stats")
        return np.array(stats['mean']), np.array(stats['std'])
    else:
        print(f"⚠ Normalization stats not found, skipping normalization")
        return None, None


def preprocess_video(video_path, preprocess_settings, mean=None, std=None):
    """
    Extract keypoints from video and preprocess
    
    Args:
        video_path: Path to video file
        preprocess_settings: Dict of preprocessing parameters
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        sequences: Tensor of shape [1, seq_len, num_joints, 3]
        joint_types: Tensor of shape [1, num_joints]
    """
    print(f"\n🎥 Processing video: {video_path}")
    
    sequence, original_length = extract_keypoints_from_video(Path(video_path), preprocess_settings)
    print(f"   Extracted {original_length} frames → processed sequence {sequence.shape}")
    
    if mean is not None and std is not None:
        sequence = normalize_keypoints(sequence, mean, std)
        print("   Applied dataset-level normalization")
    
    sequences = torch.from_numpy(sequence).unsqueeze(0)  # [1, T, J, 3]
    joint_types = get_joint_type_tensor(sequence.shape[1]).unsqueeze(0)  # [1, J]
    
    return sequences.float(), joint_types


def predict(model, sequences, joint_types, device='cuda', top_k=5):
    """
    Make prediction on sequences
    
    Args:
        model: Trained model
        sequences: Input sequences [1, T, J, 2]
        joint_types: Joint type indices [1, J]
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        predictions: List of (class_idx, probability) tuples
    """
    model.eval()
    
    sequences = sequences.to(device)
    joint_types = joint_types.to(device)
    
    with torch.no_grad():
        outputs = model(sequences, joint_types)
        probs = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k, dim=1)
        
        predictions = [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
    
    return predictions


def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    checkpoint_dir = Path('checkpoints')
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError("No checkpoints directory found!")
    
    subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError("No experiment directories found!")
    
    # Get most recent
    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
    checkpoint_path = latest_dir / 'best_model.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best_model.pth found in {latest_dir}")
    
    return checkpoint_path


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sign Language Translation Inference')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (auto-detects if not provided)')
    parser.add_argument('--data_root', type=str, default='processed',
                        help='Path to processed data directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Setup device
    requested_device = args.device
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = torch.device('cpu')
    else:
        try:
            device = torch.device(requested_device)
        except Exception as exc:
            print(f"⚠️  Invalid device '{requested_device}': {exc}. Falling back to CPU")
            device = torch.device('cpu')
    print(f"🖥️  Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        print("🔍 Auto-detecting latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        print(f"   Found: {checkpoint_path}")
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("SIGN LANGUAGE TRANSLATION - INFERENCE")
    print("="*80)
    
    # Load model
    model, model_config = load_checkpoint(checkpoint_path, device)
    
    # Load label mapping
    label_mapping = load_label_mapping(args.data_root)
    
    # Load preprocessing configuration & normalization stats
    preprocess_settings = load_preprocess_settings(args.data_root)
    mean, std = load_normalization_stats(args.data_root)
    
    # Preprocess video
    try:
        sequences, joint_types = preprocess_video(video_path, preprocess_settings, mean, std)
    except Exception as e:
        print(f"❌ Error preprocessing video: {e}")
        sys.exit(1)
    
    # Make prediction
    print(f"\n🔮 Running inference...")
    predictions = predict(model, sequences, joint_types, device, args.top_k)
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    
    for rank, (class_idx, prob) in enumerate(predictions, 1):
        if label_mapping:
            class_name = label_mapping.get(str(class_idx), f"Class {class_idx}")
        else:
            class_name = f"Class {class_idx}"
        
        confidence_bar = "█" * int(prob * 50)
        print(f"{rank}. {class_name:30s} {prob*100:6.2f}% {confidence_bar}")
    
    print("="*80)
    
    # Print top prediction prominently
    top_idx, top_prob = predictions[0]
    if label_mapping:
        top_class = label_mapping.get(str(top_idx), f"Class {top_idx}")
    else:
        top_class = f"Class {top_idx}"
    
    print(f"\n✨ TOP PREDICTION: {top_class} ({top_prob*100:.2f}% confidence)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()