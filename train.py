"""
train.py - Complete Training Pipeline for CorrFormer-Lite (FULLY FIXED)

Features:
- Mixed precision training (FP16)
- OneCycleLR scheduler with warmup (FIXED - per-batch updates)
- Label smoothing
- Gradient clipping
- Early stopping
- TensorBoard logging
- Checkpoint management
- Test evaluation with confusion matrix
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from model import create_model, count_parameters
from dataset import create_dataloader, get_dataset_statistics


# ============================================================================
# CONFIGURATION - FIXED VALUES
# ============================================================================

CONFIG = {
    # ========== Data ==========
    'data_root': 'processed',  # Path to processed data
    'num_classes': 226,                  # Number of sign classes
    
    # ========== Model ==========
    'model_size': 'base',                # 'tiny', 'small', 'base', 'large'
    'dropout': 0.1,                      # Dropout rate
    
    # ========== Training Hyperparameters (FIXED) ==========
    'batch_size': 64,                    # FIXED: Increased from 32 to 64
    'epochs': 80,                        # Maximum epochs
    'lr': 1e-4,                          # Learning rate (max LR)
    'weight_decay': 0.001,               # FIXED: Reduced from 0.01 to 0.001
    'warmup_epochs': 10,                 # FIXED: Increased from 5 to 10
    'label_smoothing': 0.05,             # FIXED: Reduced from 0.1 to 0.05
    'grad_clip': 1.0,                    # Gradient clipping norm
    
    # ========== Optimization ==========
    'mixed_precision': True,             # Use FP16 mixed precision
    'num_workers': 4,                    # DataLoader workers
    
    # ========== Augmentation ==========
    'augment': True,                     # Use data augmentation
    'aug_config': {
        'rotation_range': 5.0,           # Rotation in degrees
        'noise_std': 0.01,               # Gaussian noise std
        'temporal_mask_prob': 0.1,       # Temporal masking probability
        'temporal_mask_ratio': 0.15,     # Ratio of frames to mask
        'spatial_scale': 0.05,           # Spatial scaling range
    },
    
    # ========== Early Stopping ==========
    'early_stop_patience': 15,           # Epochs without improvement before stopping
    
    # ========== Logging & Checkpoints ==========
    'log_dir': 'logs',                   # TensorBoard logs directory
    'save_dir': 'checkpoints',           # Model checkpoints directory
    'exp_name': None,                    # Experiment name (auto-generated if None)
    
    # ========== Evaluation ==========
    'eval_only': False,                  # Only run evaluation (no training)
    'checkpoint': None,                  # Path to checkpoint for evaluation
}


def get_config():
    """Get training configuration with optional command-line overrides"""
    # Convert CONFIG dict to argparse Namespace for compatibility
    from argparse import Namespace
    
    # Start with CONFIG
    args = Namespace(**CONFIG)
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.model_size}_{timestamp}"
    
    return args


# ============================================================================
# METRICS
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


# ============================================================================
# TRAINING FUNCTIONS (FIXED - PER-BATCH SCHEDULER UPDATE)
# ============================================================================

def train_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer,
    scheduler,  # FIXED: Added scheduler for per-batch updates
    scaler, 
    device,
    epoch,
    args
):
    """Train for one epoch with per-batch scheduler updates"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (sequences, joint_types, labels) in enumerate(pbar):
        sequences = sequences.to(device, non_blocking=True)
        joint_types = joint_types.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        batch_size = sequences.size(0)
        
        # Forward pass with mixed precision
        with autocast(enabled=args.mixed_precision):
            outputs = model(sequences, joint_types)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        
        if args.mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # CRITICAL FIX: Update learning rate every batch
        scheduler.step()
        
        # Measure accuracy
        acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
        
        # Update metrics
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        
        # Update progress bar with current LR
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}%',
            'top5': f'{top5.avg:.2f}%',
            'lr': f'{current_lr:.2e}'
        })
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, args):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, joint_types, labels in tqdm(val_loader, desc="Validating"):
            sequences = sequences.to(device, non_blocking=True)
            joint_types = joint_types.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            batch_size = sequences.size(0)
            
            # Forward pass
            with autocast(enabled=args.mixed_precision):
                outputs = model(sequences, joint_types)
                loss = criterion(outputs, labels)
            
            # Measure accuracy
            acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            # Store predictions
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return losses.avg, top1.avg, top5.avg, np.array(all_preds), np.array(all_labels)


# ============================================================================
# TRAINING LOOP (FIXED)
# ============================================================================

def train(args):
    """Main training function"""
    
    # Setup directories
    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print(f"Training CorrFormer-Lite: {args.exp_name}")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Dataset statistics
    print("\n📊 Dataset Statistics:")
    stats = get_dataset_statistics(args.data_root)
    for split, split_stats in stats.items():
        print(f"  {split}: {split_stats['num_samples']} samples, "
              f"{split_stats['num_classes']} classes")
    
    # Create dataloaders
    print("\n🔄 Creating DataLoaders...")
    train_loader = create_dataloader(
        args.data_root, 'train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        aug_config=args.aug_config
    )
    
    val_loader = create_dataloader(
        args.data_root, 'val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False
    )
    
    test_loader = create_dataloader(
        args.data_root, 'test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n🏗️  Creating model: {args.model_size}")
    model = create_model(
        num_classes=args.num_classes,
        model_size=args.model_size,
        dropout=args.dropout
    )
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")
    print(f"  Size: {trainable / 1e6:.2f}M parameters")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # FIXED: Use OneCycleLR scheduler with per-batch updates
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    print(f"\n📈 Learning Rate Schedule (OneCycleLR):")
    print(f"  Initial LR: {args.lr / 25:.6f} (max_lr / 25)")
    print(f"  Max LR (after warmup): {args.lr:.6f}")
    print(f"  Final LR: {args.lr / 1000:.6f} (max_lr / 1000)")
    print(f"  Warmup steps: {warmup_steps} ({args.warmup_epochs} epochs)")
    print(f"  Total steps: {total_steps}")
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,  # Warmup percentage
        anneal_strategy='cos',
        div_factor=25.0,        # Initial LR = max_lr / 25 = 4e-6
        final_div_factor=1000.0  # Final LR = max_lr / 1000 = 1e-7
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=args.mixed_precision)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Training state
    best_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Augmentation: {args.augment}")
    print("="*80)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Get learning rate at start of epoch
        lr_start = optimizer.param_groups[0]['lr']
        
        # Train (scheduler updated per-batch inside train_epoch)
        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch + 1, args
        )
        
        # Get learning rate at end of epoch
        lr_end = optimizer.param_groups[0]['lr']
        
        # Validate
        val_loss, val_top1, val_top5, _, _ = validate(
            model, val_loader, criterion, device, args
        )
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train_top1', train_top1, epoch)
        writer.add_scalar('Accuracy/val_top1', val_top1, epoch)
        writer.add_scalar('Accuracy/train_top5', train_top5, epoch)
        writer.add_scalar('Accuracy/val_top5', val_top5, epoch)
        writer.add_scalar('LR', lr_end, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Top-1: {train_top1:.2f}% | Top-5: {train_top5:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}%")
        print(f"  LR: {lr_start:.6f} → {lr_end:.6f}")
        
        # Save checkpoint
        is_best = val_top1 > best_acc
        if is_best:
            best_acc = val_top1
            patience_counter = 0
            print(f"  🎉 New best validation accuracy: {best_acc:.2f}%")
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, save_dir / 'best_model.pth')
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.early_stop_patience}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Always save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'args': vars(args)
        }, save_dir / 'last_model.pth')
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            break
    
    writer.close()
    
    print("\n" + "="*80)
    print("✅ Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print("="*80)
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_top1, test_top5, test_preds, test_labels = validate(
        model, test_loader, criterion, device, args
    )
    
    print(f"\n✨ Final Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1 Accuracy: {test_top1:.2f}%")
    print(f"  Top-5 Accuracy: {test_top5:.2f}%")
    
    # Save test results
    results = {
        'test_loss': test_loss,
        'test_top1': test_top1,
        'test_top5': test_top5,
        'best_val_acc': best_acc,
        'training_epochs': epoch + 1
    }
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate confusion matrix
    print("\n📈 Generating confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds, save_dir)
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(test_labels, test_preds, args.num_classes)
    
    # Save per-class results
    df = pd.DataFrame({
        'class': range(args.num_classes),
        'accuracy': per_class_acc
    })
    df.to_csv(save_dir / 'per_class_accuracy.csv', index=False)
    
    # Print best and worst classes
    print("\n🏆 Best performing classes:")
    best_classes = df.nlargest(10, 'accuracy')
    for idx, row in best_classes.iterrows():
        print(f"  Class {int(row['class'])}: {row['accuracy']:.2f}%")
    
    print("\n🔍 Worst performing classes:")
    worst_classes = df.nsmallest(10, 'accuracy')
    for idx, row in worst_classes.iterrows():
        print(f"  Class {int(row['class'])}: {row['accuracy']:.2f}%")
    
    print("\n" + "="*80)
    print(f"All results saved to: {save_dir}")
    print("="*80)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(labels, preds, save_dir, top_n=50):
    """Plot confusion matrix for top N classes"""
    from sklearn.metrics import confusion_matrix
    
    # Get top N most frequent classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_classes = unique_labels[np.argsort(counts)[-top_n:]]
    
    # Filter to top classes
    mask = np.isin(labels, top_classes)
    filtered_labels = labels[mask]
    filtered_preds = preds[mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes)
    
    # Normalize
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, cmap='Blues', cbar_kws={'label': 'Accuracy (%)'}, 
                vmin=0, vmax=100, square=True)
    plt.title(f'Confusion Matrix (Top {top_n} Most Frequent Classes)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved confusion matrix: {save_dir / 'confusion_matrix.png'}")


def compute_per_class_accuracy(labels, preds, num_classes):
    """Compute per-class accuracy"""
    per_class_acc = []
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            acc = (preds[mask] == c).mean() * 100
        else:
            acc = 0.0
        per_class_acc.append(acc)
    
    return per_class_acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if args.eval_only:
        # TODO: Implement evaluation only mode
        raise NotImplementedError("Evaluation only mode not yet implemented")
    else:
        train(args)


if __name__ == '__main__':
    main()