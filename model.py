"""
model.py - CorrFormer-Lite for Sign Language Recognition

Architecture:
1. Joint-level Semantic Embeddings (inspired by CorrFormer)
2. Spatial Encoder: Self-Attention over joints within each frame
3. Temporal Encoder: Transformer over frame sequence
4. Classification Head with attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal modeling"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x + positional encoding: (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class JointEmbedding(nn.Module):
    """
    Joint-level semantic embedding
    Projects (x, y, confidence) + joint_type to d_model
    """
    
    def __init__(self, d_model: int, num_joint_types: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Coordinate projection: (x, y, conf) -> d_model
        self.coord_projection = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Semantic type embedding (pose=0, left_hand=1, right_hand=2)
        self.type_embedding = nn.Embedding(num_joint_types, d_model)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, coords: torch.Tensor, joint_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, T, J, 3) - x, y, confidence
            joint_types: (B, J) - semantic type ID per joint
        Returns:
            embeddings: (B, T, J, d_model)
        """
        B, T, J, _ = coords.shape
        
        # Project coordinates
        coord_emb = self.coord_projection(coords)  # (B, T, J, d_model)
        
        # Add semantic type embeddings
        type_emb = self.type_embedding(joint_types)  # (B, J, d_model)
        type_emb = type_emb.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, J, d_model)
        
        # Combine
        embeddings = coord_emb + type_emb
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class SpatialEncoder(nn.Module):
    """
    Spatial encoder: models correlations between joints within each frame
    Uses multi-head self-attention
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int = 4, 
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Frame-level pooling (attention-based)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, d_model) - joint embeddings per frame
        Returns:
            frame_features: (B, T, d_model) - frame-level features
        """
        B, T, J, D = x.shape
        
        # Process each frame independently
        # Reshape: (B*T, J, d_model)
        x_flat = x.view(B * T, J, D)
        
        # Apply spatial attention across joints
        spatial_features = self.encoder(x_flat)  # (B*T, J, d_model)
        
        # Pool joints to get frame representation
        # Option 1: Mean pooling
        # frame_features = spatial_features.mean(dim=1)  # (B*T, d_model)
        
        # Option 2: Attention pooling (better)
        attention_weights = self.attention_pool(spatial_features)  # (B*T, J, 1)
        frame_features = (spatial_features * attention_weights).sum(dim=1)  # (B*T, d_model)
        
        # Reshape back: (B, T, d_model)
        frame_features = frame_features.view(B, T, D)
        
        return frame_features


class TemporalEncoder(nn.Module):
    """
    Temporal encoder: models motion across frames
    Uses Transformer with positional encoding
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128
    ):
        super().__init__()
        
        # Positional encoding for temporal order
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) - frame-level features
            mask: Optional attention mask for padding
        Returns:
            temporal_features: (B, T, d_model)
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply temporal attention
        temporal_features = self.encoder(x, src_key_padding_mask=mask)
        
        return temporal_features


class AttentionPooling(nn.Module):
    """Learnable attention pooling for sequence aggregation"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            pooled: (B, d_model)
        """
        # Compute attention weights
        weights = self.attention(x)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (B, d_model)
        
        return pooled


class CorrFormerLite(nn.Module):
    """
    CorrFormer-Lite: Lightweight architecture for sign language recognition
    
    Pipeline:
    1. Joint Embeddings (coordinates + semantic types)
    2. Spatial Encoder (joint correlations within frames)
    3. Temporal Encoder (motion across frames)
    4. Classification Head
    
    Args:
        num_classes: Number of sign classes (226 for AUTSL)
        num_joints: Number of joints per frame (56)
        num_joint_types: Number of semantic joint types (3: pose, left_hand, right_hand)
        d_model: Model dimension (128 recommended)
        spatial_layers: Number of spatial attention layers
        temporal_layers: Number of temporal attention layers
        nhead_spatial: Number of attention heads for spatial
        nhead_temporal: Number of attention heads for temporal
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 226,
        num_joints: int = 56,
        num_joint_types: int = 3,
        d_model: int = 128,
        spatial_layers: int = 2,
        temporal_layers: int = 4,
        nhead_spatial: int = 4,
        nhead_temporal: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.d_model = d_model
        
        # 1. Joint-level embeddings
        self.joint_embedding = JointEmbedding(
            d_model=d_model,
            num_joint_types=num_joint_types,
            dropout=dropout
        )
        
        # 2. Spatial encoder (frame-level)
        self.spatial_encoder = SpatialEncoder(
            d_model=d_model,
            nhead=nhead_spatial,
            num_layers=spatial_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        # 3. Temporal encoder (sequence-level)
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            nhead=nhead_temporal,
            num_layers=temporal_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        # 4. Pooling
        self.pooling = AttentionPooling(d_model)
        
        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        sequences: torch.Tensor, 
        joint_types: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            sequences: (B, T, J, 3) - pose sequences
            joint_types: (B, J) - semantic joint type IDs
            return_features: If True, return features before classification
        
        Returns:
            logits: (B, num_classes) or features: (B, d_model)
        """
        B, T, J, C = sequences.shape
        
        # 1. Joint embeddings
        joint_emb = self.joint_embedding(sequences, joint_types)  # (B, T, J, d_model)
        
        # 2. Spatial encoding (frame-level joint correlations)
        frame_features = self.spatial_encoder(joint_emb)  # (B, T, d_model)
        
        # 3. Temporal encoding (motion modeling)
        temporal_features = self.temporal_encoder(frame_features)  # (B, T, d_model)
        
        # 4. Global pooling
        sequence_features = self.pooling(temporal_features)  # (B, d_model)
        
        if return_features:
            return sequence_features
        
        # 5. Classification
        logits = self.classifier(sequence_features)  # (B, num_classes)
        
        return logits


def create_model(
    num_classes: int = 226,
    model_size: str = 'base',
    dropout: float = 0.1
) -> CorrFormerLite:
    """
    Factory function to create CorrFormer-Lite models of different sizes
    
    Args:
        num_classes: Number of classes
        model_size: 'tiny', 'small', 'base', or 'large'
        dropout: Dropout rate
    
    Returns:
        CorrFormerLite model
    """
    configs = {
        'tiny': {
            'd_model': 64,
            'spatial_layers': 1,
            'temporal_layers': 2,
            'nhead_spatial': 4,
            'nhead_temporal': 4,
        },
        'small': {
            'd_model': 96,
            'spatial_layers': 2,
            'temporal_layers': 3,
            'nhead_spatial': 4,
            'nhead_temporal': 6,
        },
        'base': {
            'd_model': 128,
            'spatial_layers': 2,
            'temporal_layers': 4,
            'nhead_spatial': 4,
            'nhead_temporal': 8,
        },
        'large': {
            'd_model': 192,
            'spatial_layers': 3,
            'temporal_layers': 6,
            'nhead_spatial': 6,
            'nhead_temporal': 12,
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    model = CorrFormerLite(
        num_classes=num_classes,
        num_joints=56,
        num_joint_types=3,
        dropout=dropout,
        **config
    )
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test the model"""
    
    print("="*80)
    print("Testing CorrFormer-Lite Model")
    print("="*80)
    
    # Test configurations
    batch_size = 8
    seq_len = 64
    num_joints = 56
    num_classes = 226
    
    # Create dummy data
    sequences = torch.randn(batch_size, seq_len, num_joints, 3)
    joint_types = torch.randint(0, 3, (batch_size, num_joints))
    
    print(f"\nInput shapes:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Joint types: {joint_types.shape}")
    
    # Test different model sizes
    for size in ['tiny', 'small', 'base', 'large']:
        print(f"\n{'='*60}")
        print(f"Testing {size.upper()} model")
        print(f"{'='*60}")
        
        model = create_model(num_classes=num_classes, model_size=size)
        model.eval()
        
        # Count parameters
        trainable, total = count_parameters(model)
        print(f"  Parameters: {trainable:,} trainable / {total:,} total")
        print(f"  Size: {trainable / 1e6:.2f}M parameters")
        
        # Forward pass
        with torch.no_grad():
            logits = model(sequences, joint_types)
        
        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Test with features
        features = model(sequences, joint_types, return_features=True)
        print(f"  Feature shape: {features.shape}")
        
        # Memory estimate
        param_memory = trainable * 4 / 1024 / 1024  # FP32 in MB
        print(f"  Memory (params): ~{param_memory:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ All model tests passed!")
    print("="*80)
    
    # Recommended configuration
    print("\n📋 Recommended Configuration for RTX 4050:")
    print("  Model size: 'base' (1.4M params)")
    print("  Batch size: 16-32")
    print("  Mixed precision: FP16")
    print("  Expected VRAM: ~2-3 GB")