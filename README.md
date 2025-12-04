# Sign Language Recognition with CorrFormer-Lite

A state-of-the-art sign language recognition system using the **CorrFormer-Lite** architecture, trained on the AUTSL (Turkish Sign Language) dataset. This project achieves high accuracy through advanced pose-based feature extraction and transformer-based temporal modeling.

## 🌟 Features

- **CorrFormer-Lite Architecture**: Lightweight transformer model with correlation-based spatial pooling
- **Rotary Position Embeddings (RoPE)**: Enhanced temporal modeling for motion sequences
- **MediaPipe Integration**: Robust pose and hand landmark extraction
- **Multiple Model Sizes**: Tiny, Small, Base, and Large variants (0.5M - 4M parameters)
- **Advanced Training**: Mixed precision, EMA, gradient accumulation, and OneCycleLR scheduling
- **Data Augmentation**: Rotation, scaling, flipping, temporal masking, and noise injection
- **Real-time Inference**: Single video prediction with confidence scores

## 📊 Model Performance

| Model Size | Parameters | Test Accuracy (Top-1) | Test Accuracy (Top-5) |
|------------|------------|----------------------|----------------------|
| Tiny       | ~0.5M      | TBD                  | TBD                  |
| Small      | ~1.2M      | TBD                  | TBD                  |
| Base       | ~2.5M      | TBD                  | TBD                  |
| Large      | ~4.0M      | TBD                  | TBD                  |

*Note: Update these values after training completion*

## 🏗️ Architecture Overview

### CorrFormer-Lite Components

1. **Joint Embedding Layer**
   - Converts raw keypoints (x, y, confidence) to high-dimensional embeddings
   - Semantic type embeddings for pose, left hand, and right hand joints
   - Single LayerNorm for efficiency

2. **Spatial Encoder**
   - Models joint-to-joint correlations within each frame
   - Correlation-based pooling for frame-level features
   - Joint-level positional embeddings

3. **Temporal Encoder**
   - Captures motion patterns across frames
   - RoPE (Rotary Position Embeddings) for better temporal modeling
   - Multi-head self-attention with pre-normalization

4. **Classification Head**
   - Attention-based pooling for sequence aggregation
   - Two-layer MLP with GELU activation
   - 226 output classes (AUTSL dataset)

### Key Innovations

- **Correlation Pooling**: Captures spatial dependencies between joints
- **RoPE**: Improved positional encoding for temporal sequences
- **DropPath**: Stochastic depth for better regularization
- **Joint-level Semantics**: Separate embeddings for pose, left hand, right hand

## 📁 Project Structure

```
SLT/
├── model.py                    # CorrFormer-Lite model architecture
├── dataset.py                  # PyTorch dataset and data loading
├── train.py                    # Training pipeline with optimizations
├── inference.py                # Single video inference script
├── preprocessing.py            # MediaPipe keypoint extraction
├── generateplots.py            # Training visualization and analysis
├── dataset_analysis.py         # Dataset statistics and exploration
├── checkpoints/                # Saved model checkpoints
│   ├── base_optimized_*/       # Base model checkpoints
│   │   ├── best_model.pth      # Best validation accuracy model
│   │   ├── config.json         # Training configuration
│   │   └── test_results.json   # Final test results
│   └── ...
├── logs/                       # TensorBoard training logs
├── processed_autsl/            # Preprocessed dataset
│   ├── train/                  # Training sequences (.npz files)
│   ├── val/                    # Validation sequences
│   ├── test/                   # Test sequences
│   ├── train_metadata.csv      # Training metadata
│   ├── val_metadata.csv        # Validation metadata
│   └── test_metadata.csv       # Test metadata
├── autsl/                      # Raw AUTSL dataset (videos)
├── hand_landmarker.task        # MediaPipe hand model
├── pose_landmarker_lite.task   # MediaPipe pose model
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA 11.8+ (for GPU training)

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas opencv-python mediapipe tqdm tensorboard matplotlib seaborn scikit-learn
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd SLT
   ```

2. **Download MediaPipe models** (if not already present)
   - `hand_landmarker.task` - Hand landmark detection
   - `pose_landmarker_lite.task` - Pose landmark detection
   
   Download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

3. **Prepare the dataset**
   - Download AUTSL dataset and place videos in `autsl/` directory
   - Run preprocessing:
     ```bash
     python preprocessing.py
     ```

## 🎯 Usage

### Training

Train a model from scratch:

```bash
python train.py
```

The training script will:
- Load preprocessed data from `processed_autsl/`
- Create a new experiment directory in `checkpoints/`
- Save the best model based on validation accuracy
- Log training metrics to TensorBoard

**Training Configuration** (edit in `train.py`):

```python
CONFIG = {
    'model_size': 'base',        # 'tiny', 'small', 'base', 'large'
    'batch_size': 64,            # Adjust based on GPU memory
    'epochs': 80,
    'lr': 5e-4,
    'dropout': 0.1,
    'mixed_precision': True,     # Use AMP for faster training
    'use_ema': True,             # Exponential Moving Average
    'gradient_accumulation': 2,  # Effective batch size = 128
    'augment': True,             # Enable data augmentation
}
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in your browser to see:
- Training/validation loss curves
- Top-1 and Top-5 accuracy
- Learning rate schedule
- Gradient norms

### Inference

Run inference on a single video:

```bash
# Auto-detect latest checkpoint
python inference.py --video path/to/video.mp4

# Use specific checkpoint
python inference.py --video path/to/video.mp4 --checkpoint checkpoints/base_optimized_*/best_model.pth

# Adjust top-k predictions
python inference.py --video path/to/video.mp4 --top_k 10
```

**Example Output:**

```
================================================================================
PREDICTIONS
================================================================================
1. Hello                        95.23% ██████████████████████████████████████████████████
2. Thank you                     2.15% ██
3. Good morning                  1.32% █
4. Yes                           0.89% 
5. No                            0.41% 

✨ TOP PREDICTION: Hello (95.23% confidence)
================================================================================
```

### Evaluation

Evaluate model on test set:

```bash
python train.py --eval_only --checkpoint checkpoints/base_optimized_*/best_model.pth
```

## 📊 Dataset

### AUTSL (Turkish Sign Language)

- **Classes**: 226 sign words
- **Samples**: ~38,000 videos
- **Splits**: Train (70%), Validation (15%), Test (15%)
- **Format**: RGB videos at 30 FPS
- **Signers**: Multiple signers with varying backgrounds

### Preprocessing Pipeline

1. **Frame Sampling**: Uniformly sample up to 64 frames per video
2. **Keypoint Extraction**: 
   - 14 upper body pose landmarks
   - 21 left hand landmarks
   - 21 right hand landmarks
   - Total: 56 keypoints per frame
3. **Normalization**: Per-frame centering and scaling
4. **Storage**: Compressed `.npz` format for fast loading

### Data Augmentation

Applied during training:
- **Rotation**: ±5° in XY plane
- **Scaling**: ±5% spatial scaling
- **Horizontal Flip**: 20% probability (with hand swapping)
- **Temporal Masking**: 10% probability, 15% of frames
- **Gaussian Noise**: σ=0.01

## 🔧 Model Configuration

### Available Model Sizes

```python
# Tiny - Fast inference, lower accuracy
create_model(num_classes=226, model_size='tiny')
# Parameters: ~500K, d_model=64

# Small - Balanced performance
create_model(num_classes=226, model_size='small')
# Parameters: ~1.2M, d_model=96

# Base - Recommended for most use cases
create_model(num_classes=226, model_size='base')
# Parameters: ~2.5M, d_model=128

# Large - Maximum accuracy
create_model(num_classes=226, model_size='large')
# Parameters: ~4.0M, d_model=192
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 5e-4 | Peak LR with OneCycleLR |
| Weight Decay | 0.01 | L2 regularization |
| Warmup Epochs | 5 | Linear warmup period |
| Label Smoothing | 0.1 | Prevents overconfidence |
| Dropout | 0.1 | Applied in embeddings and classifier |
| DropPath | 0.15 | Stochastic depth (base model) |
| Gradient Clip | 1.0 | Max gradient norm |

## 📈 Training Tips

### GPU Memory Optimization

If you encounter OOM errors:

1. **Reduce batch size**: `batch_size=32` or `batch_size=16`
2. **Use gradient accumulation**: `gradient_accumulation=4` (effective batch size = batch_size × 4)
3. **Use smaller model**: `model_size='small'` or `model_size='tiny'`
4. **Disable EMA**: `use_ema=False`

### Improving Accuracy

1. **Increase model capacity**: Use `model_size='large'`
2. **Train longer**: `epochs=100` or `epochs=120`
3. **Stronger augmentation**: Increase augmentation parameters
4. **Ensemble models**: Train multiple models and average predictions
5. **Fine-tune on specific signs**: Use transfer learning for specific classes

### Faster Training

1. **Enable mixed precision**: `mixed_precision=True` (default)
2. **Use fused optimizer**: `use_fused_optimizer=True` (CUDA only)
3. **Increase workers**: `num_workers=8` (adjust based on CPU cores)
4. **Enable persistent workers**: `persistent_workers=True`
5. **Use torch.compile**: `use_compile=True` (PyTorch 2.0+, Linux only)

## 🐛 Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
- Reduce batch size or use gradient accumulation
- Close other GPU applications
- Use a smaller model size

**2. "All-zero sequences in preprocessing"**
- Check MediaPipe model files are present
- Verify video files are valid and readable
- Ensure sufficient lighting in videos

**3. "NaN loss during training"**
- Reduce learning rate: `lr=1e-4`
- Check data for NaN/Inf values
- Disable mixed precision temporarily

**4. "Low validation accuracy"**
- Train for more epochs
- Increase model size
- Enable data augmentation
- Check data preprocessing

**5. "Inference gives wrong predictions"**
- Ensure video quality is good
- Check that preprocessing settings match training
- Verify checkpoint is loaded correctly

## 📝 File Descriptions

### Core Files

- **`model.py`**: CorrFormer-Lite architecture implementation
  - `CorrFormerLite`: Main model class
  - `create_model()`: Factory function for different model sizes
  - `RotaryPositionEmbedding`: RoPE implementation
  - `SpatialEncoder`: Frame-level joint correlation modeling
  - `TemporalEncoder`: Motion sequence modeling

- **`dataset.py`**: Data loading and augmentation
  - `PoseSignDataset`: PyTorch dataset class
  - `create_dataloader()`: DataLoader factory with optimizations
  - Data augmentation functions
  - Normalization utilities

- **`train.py`**: Training pipeline
  - Mixed precision training (AMP)
  - EMA (Exponential Moving Average)
  - OneCycleLR scheduling
  - Early stopping
  - TensorBoard logging

- **`inference.py`**: Single video prediction
  - Video frame sampling
  - MediaPipe keypoint extraction
  - Preprocessing pipeline
  - Top-k prediction display

- **`preprocessing.py`**: Dataset preprocessing
  - MediaPipe pose and hand detection
  - Keypoint extraction and normalization
  - Metadata generation
  - Parallel processing support

### Utility Files

- **`generateplots.py`**: Training visualization
  - Loss and accuracy curves
  - Confusion matrices
  - Per-class performance analysis

- **`dataset_analysis.py`**: Dataset exploration
  - Class distribution
  - Sequence length statistics
  - Data quality checks

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{corrformer-lite-slt,
  title={CorrFormer-Lite: Lightweight Sign Language Recognition with Correlation-based Transformers},
  author={Aryan Vohra},
  year={2025},
  howpublished={\url{https://github.com/Tengenisu/sign_language_recognition/}}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AUTSL Dataset**: Turkish Sign Language dataset
- **MediaPipe**: Google's pose and hand landmark detection
- **PyTorch**: Deep learning framework
- **CorrFormer**: Original correlation-based transformer architecture

## 🔗 References

1. [AUTSL Dataset Paper](https://arxiv.org/abs/2008.00932)
2. [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
3. [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864)
4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

**Happy Signing! 👋**
