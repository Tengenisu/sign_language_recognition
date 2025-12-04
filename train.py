"""
train_optimized.py - Fixed Training Pipeline for CorrFormer-Lite

FULLY FIXED VERSION:
✅ Correctly unpacks 3-tuple from dataloader (sequences, labels, joint_types)
✅ Removed joint_types from function parameters (comes from dataloader now)
✅ Proper device transfer for all tensors
✅ torch.compile() disabled by default (Windows/Triton issue)
✅ All optimizations preserved
✅ EMA support
✅ Mixed precision training
✅ Gradient accumulation
"""

import os
import json
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

from model import create_model, count_parameters
from dataset import create_dataloader, get_dataset_statistics


class EMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'running_mean' not in name and 'running_var' not in name:
                    self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


CONFIG = {
    'data_root': 'processed_autsl',
    'num_classes': 226,
    'model_size': 'large', #Options: 'tiny', 'small', 'base', 'large'
    'dropout': 0.1,
    'batch_size': 64,
    'epochs': 80,
    'lr': 5e-4,  # FIXED: Increased from 1e-4 (was too low)
    'weight_decay': 0.01,  # FIXED: Increased for better regularization
    'warmup_epochs': 5,  # FIXED: Reduced warmup (was too long)
    'label_smoothing': 0.1,  # FIXED: Increased slightly
    'grad_clip': 1.0,
    'mixed_precision': True,
    'num_workers': 4,
    'use_compile': False,  # Disabled by default (Windows incompatible)
    'use_ema': True,
    'ema_decay': 0.999,
    'use_balanced_sampling': False,
    'gradient_accumulation': 2,  # FIXED: Increased for effective batch size of 128
    'use_fused_optimizer': True,
    'augment': True,
    'aug_config': {
        'rotation_range': 5.0,
        'noise_std': 0.01,
        'temporal_mask_prob': 0.1,
        'temporal_mask_ratio': 0.15,
        'spatial_scale': 0.05,
    },
    'early_stop_patience': 15,
    'log_dir': 'logs',
    'save_dir': 'checkpoints',
    'exp_name': None,
    'eval_only': False,
    'checkpoint': None,
}


def get_config():
    """Get training configuration"""
    from argparse import Namespace
    args = Namespace(**CONFIG)
    
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.model_size}_optimized_{timestamp}"
    
    return args


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


def train_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer,
    scheduler,
    scaler, 
    device,
    epoch,
    args,
    ema=None
):
    """
    Train for one epoch - FULLY FIXED
    
    FIXED: Now correctly unpacks 3-tuple from dataloader
    """
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    accum_steps = args.gradient_accumulation
    
    # FIXED: Unpack 3 values (sequences, labels, joint_types)
    for batch_idx, (sequences, labels, joint_types) in enumerate(pbar):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        joint_types = joint_types.to(device, non_blocking=True)  # FIXED: Added
        batch_size = sequences.size(0)
        
        # Expand joint_types to match batch size
        batch_joint_types = joint_types.unsqueeze(0).expand(batch_size, -1)
        
        with autocast(enabled=args.mixed_precision):
            outputs = model(sequences, batch_joint_types)
            loss = criterion(outputs, labels)
            
            if accum_steps > 1:
                loss = loss / accum_steps
        
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            if args.mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            if ema is not None:
                ema.update()
        
        with torch.no_grad():
            acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
            actual_loss = loss.item() * accum_steps if accum_steps > 1 else loss.item()
            losses.update(actual_loss, batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}%',
            'top5': f'{top5.avg:.2f}%',
            'lr': f'{current_lr:.2e}'
        })
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, args):
    """
    Validate the model - FULLY FIXED
    
    FIXED: Now correctly unpacks 3-tuple from dataloader
    """
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # FIXED: Unpack 3 values (sequences, labels, joint_types)
        for sequences, labels, joint_types in tqdm(val_loader, desc="Validating"):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            joint_types = joint_types.to(device, non_blocking=True)  # FIXED: Added
            batch_size = sequences.size(0)
            
            # Expand joint_types to match batch size
            batch_joint_types = joint_types.unsqueeze(0).expand(batch_size, -1)
            
            with autocast(enabled=args.mixed_precision):
                outputs = model(sequences, batch_joint_types)
                loss = criterion(outputs, labels)
            
            acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
            
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return losses.avg, top1.avg, top5.avg, np.array(all_preds), np.array(all_labels)


def train(args):
    """Main training function - FULLY FIXED"""
    
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled")
    
    # Create directories
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset statistics
    print("\n📊 Dataset Statistics:")
    stats = get_dataset_statistics(args.data_root)
    for split, split_stats in stats.items():
        print(f"  {split}: {split_stats['num_samples']} samples, {split_stats['num_classes']} classes")
    
    # Create dataloaders
    print("\n🔄 Creating DataLoaders...")
    train_loader = create_dataloader(
        args.data_root, 'train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        aug_config=args.aug_config,
        normalize=False,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = create_dataloader(
        args.data_root, 'val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        normalize=False,  # CRITICAL FIX: Data already normalized in preprocessing!
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    test_loader = create_dataloader(
        args.data_root, 'test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        normalize=False,  # CRITICAL FIX: Data already normalized in preprocessing!
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # CRITICAL: Data sanity check
    print("\n🔬 Data Sanity Check:")
    sample_sequences, sample_labels, sample_joint_types = next(iter(train_loader))
    print(f"  Batch shapes: sequences={sample_sequences.shape}, labels={sample_labels.shape}, joint_types={sample_joint_types.shape}")
    print(f"  Sequences stats:")
    print(f"    X: min={sample_sequences[:,:,:,0].min():.4f}, max={sample_sequences[:,:,:,0].max():.4f}, mean={sample_sequences[:,:,:,0].mean():.4f}")
    print(f"    Y: min={sample_sequences[:,:,:,1].min():.4f}, max={sample_sequences[:,:,:,1].max():.4f}, mean={sample_sequences[:,:,:,1].mean():.4f}")
    print(f"    Z: min={sample_sequences[:,:,:,2].min():.4f}, max={sample_sequences[:,:,:,2].max():.4f}, mean={sample_sequences[:,:,:,2].mean():.4f}")
    print(f"  Non-zero percentage: {(sample_sequences != 0).float().mean()*100:.2f}%")
    print(f"  Has NaN: {torch.isnan(sample_sequences).any()}")
    print(f"  Has Inf: {torch.isinf(sample_sequences).any()}")
    print(f"  Label range: {sample_labels.min()}-{sample_labels.max()}")
    
    if torch.isnan(sample_sequences).any() or torch.isinf(sample_sequences).any():
        raise ValueError("❌ Data contains NaN or Inf! Check preprocessing.")
    
    if (sample_sequences != 0).float().mean() < 0.1:
        print("  ⚠️  WARNING: Most data is zeros! Check preprocessing.")
    else:
        print("  ✅ Data looks healthy!")
    
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
    print(f"  Model size: {trainable / 1e6:.2f}M parameters")
    
    # Initialize EMA
    ema = None
    if args.use_ema:
        print(f"\n📈 Initializing EMA (decay={args.ema_decay})")
        ema = EMA(model, decay=args.ema_decay)
    
    # Compile model (if enabled and available)
    if args.use_compile and hasattr(torch, 'compile'):
        print("\n🚀 Attempting to compile model...")
        try:
            import platform
            if platform.system() == 'Windows':
                print("  ⚠️  torch.compile() not fully supported on Windows")
                print("  Skipping compilation (training will continue normally)")
            else:
                model = torch.compile(model)
                print("  ✅ Model compiled successfully")
        except Exception as e:
            print(f"  ⚠️ Compilation failed: {e}")
            print("  Continuing without compilation")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer
    optimizer_kwargs = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'betas': (0.9, 0.999)
    }
    
    if args.use_fused_optimizer and torch.cuda.is_available():
        try:
            optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs, fused=True)
            print("\n⚡ Using fused AdamW optimizer")
        except:
            optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
            print("\n📊 Using standard AdamW optimizer")
    else:
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
        print("\n📊 Using standard AdamW optimizer")
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(
        enabled=args.mixed_precision,
        init_scale=2**12,
        growth_interval=100
    )
    
    if args.mixed_precision:
        print("  ✅ Mixed precision training enabled")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training state
    best_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\n📂 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        if 'ema_shadow' in checkpoint and ema is not None:
            ema.shadow = checkpoint['ema_shadow']
        print(f"  Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("="*80)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            scaler, device, epoch + 1, args, ema
        )
        
        # Apply EMA for validation
        if ema is not None:
            ema.apply_shadow()
        
        # Validate
        val_loss, val_top1, val_top5, _, _ = validate(
            model, val_loader, criterion, device, args
        )
        
        # Restore original parameters
        if ema is not None:
            ema.restore()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train_top1', train_top1, epoch)
        writer.add_scalar('Accuracy/val_top1', val_top1, epoch)
        writer.add_scalar('Accuracy/train_top5', train_top5, epoch)
        writer.add_scalar('Accuracy/val_top5', val_top5, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train: Loss {train_loss:.4f} | Top-1 {train_top1:.2f}% | Top-5 {train_top5:.2f}%")
        print(f"  Val:   Loss {val_loss:.4f} | Top-1 {val_top1:.2f}% | Top-5 {val_top5:.2f}%")
        
        # Save best model
        is_best = val_top1 > best_acc
        if is_best:
            best_acc = val_top1
            patience_counter = 0
            print(f"  🎉 New best validation accuracy: {best_acc:.2f}%")
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }
            if ema is not None:
                save_dict['ema_shadow'] = ema.shadow
            
            torch.save(save_dict, save_dir / 'best_model.pth')
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.early_stop_patience}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }
            if ema is not None:
                save_dict['ema_shadow'] = ema.shadow
            torch.save(save_dict, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  💾 Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
            break
    
    writer.close()
    
    print("\n" + "="*80)
    print(f"✅ Training completed! Best validation accuracy: {best_acc:.2f}%")
    print("="*80)
    
    # Final evaluation on test set
    print("\n📊 Evaluating on test set with best model...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'ema_shadow' in checkpoint and ema is not None:
        ema.shadow = checkpoint['ema_shadow']
        ema.apply_shadow()
    
    test_loss, test_top1, test_top5, test_preds, test_labels = validate(
        model, test_loader, criterion, device, args
    )
    
    print(f"\n✨ Final Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1 Accuracy: {test_top1:.2f}%")
    print(f"  Top-5 Accuracy: {test_top5:.2f}%")
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_top1': float(test_top1),
        'test_top5': float(test_top5),
        'best_val_acc': float(best_acc),
    }
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All results saved to: {save_dir}")
    print("="*80)


def main():
    """Main entry point"""
    args = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("🌱 Random seeds set for reproducibility")
    
    train(args)


if __name__ == '__main__':
    main()