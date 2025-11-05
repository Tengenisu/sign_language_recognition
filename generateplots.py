"""
generate_plots.py - Comprehensive Evaluation and Visualization

Generates all plots and analysis for a trained model:
- Confusion matrix
- Per-class metrics (accuracy, precision, recall, F1)
- Macro/Micro/Weighted averages
- Top-K accuracy curves
- Error analysis
- Confidence distribution
- Training history plots (if available)

Usage:
    Just configure the CONFIG section below and run:
    python generate_plots.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from model import create_model
from dataset import create_dataloader


# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

CONFIG = {
    # ========== Paths ==========
    'exp_name': None,  # Experiment name (e.g., 'base_20251025_132443')
                       # Set to None to auto-detect most recent experiment
    
    'checkpoint_path': None,  # Or specify full checkpoint path
                              # E.g., 'checkpoints/base_20251025_132443/best_model.pth'
                              # If specified, this overrides exp_name
    
    'data_root': 'processed',  # Path to processed data
    
    'output_dir': 'model_plots',  # Output directory (default: same as checkpoint dir)
                         # E.g., 'analysis_results' or None for default
    
    # ========== Evaluation Settings ==========
    'batch_size': 64,        # Batch size for evaluation
    'num_workers': 4,        # Number of data loading workers
    'device': 'cuda',        # 'cuda' or 'cpu'
    
    # ========== Plot Settings ==========
    'confusion_matrix_top_n': 50,   # Number of classes to show in confusion matrix
    'error_analysis_top_n': 20,     # Number of top errors to analyze
    'topk_max': 20,                 # Maximum k for top-k accuracy curve
    'dpi': 150,                     # Plot resolution
    'figsize_large': (16, 12),      # Figure size for multi-plot figures
    'figsize_medium': (12, 8),      # Figure size for medium plots
    'figsize_small': (10, 6),       # Figure size for single plots
}


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, num_classes=226):
    """
    Comprehensive evaluation of the model
    
    Returns:
        results: Dict with predictions, labels, probabilities, etc.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_top5_preds = []
    
    print("🔍 Running evaluation...")
    with torch.no_grad():
        for sequences, joint_types, labels in tqdm(dataloader, desc="Evaluating"):
            sequences = sequences.to(device, non_blocking=True)
            joint_types = joint_types.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(sequences, joint_types)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = outputs.max(1)
            _, top5_preds = outputs.topk(5, 1, True, True)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_top5_preds.extend(top5_preds.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_top5_preds = np.array(all_top5_preds)
    
    # Compute metrics
    correct = (all_preds == all_labels)
    top1_acc = correct.mean() * 100
    
    # Top-5 accuracy
    top5_correct = np.any(all_top5_preds == all_labels.reshape(-1, 1), axis=1)
    top5_acc = top5_correct.mean() * 100
    
    # Confidence scores
    pred_confidences = all_probs[np.arange(len(all_preds)), all_preds]
    true_confidences = all_probs[np.arange(len(all_labels)), all_labels]
    
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'top5_predictions': all_top5_preds,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'correct': correct,
        'pred_confidences': pred_confidences,
        'true_confidences': true_confidences,
    }
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"   Top-5 Accuracy: {top5_acc:.2f}%")
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(results, output_dir, config):
    """Plot confusion matrix for top N most frequent classes"""
    labels = results['labels']
    preds = results['predictions']
    top_n = config['confusion_matrix_top_n']
    
    print(f"\n📊 Generating confusion matrix (top {top_n} classes)...")
    
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
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Raw counts
    sns.heatmap(cm, cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'},
                square=True, xticklabels=False, yticklabels=False)
    axes[0].set_title(f'Confusion Matrix - Counts (Top {top_n} Classes)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_norm, cmap='Blues', ax=axes[1], cbar_kws={'label': 'Accuracy (%)'},
                vmin=0, vmax=100, square=True, xticklabels=False, yticklabels=False)
    axes[1].set_title(f'Confusion Matrix - Normalized (Top {top_n} Classes)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: confusion_matrix.png")


def plot_per_class_accuracy(results, output_dir, config, num_classes=226):
    """Plot per-class accuracy analysis with precision, recall, F1"""
    labels = results['labels']
    preds = results['predictions']
    
    print(f"\n📊 Analyzing per-class metrics (accuracy, precision, recall, F1)...")
    
    # Compute per-class metrics
    per_class_stats = []
    
    for c in range(num_classes):
        # True Positives, False Positives, False Negatives
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        
        num_samples = (labels == c).sum()
        
        # Accuracy
        if num_samples > 0:
            accuracy = tp / num_samples * 100
        else:
            accuracy = 0.0
        
        # Precision
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0.0
        
        # Recall
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0.0
        
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        per_class_stats.append({
            'class': c,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': num_samples,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    df = pd.DataFrame(per_class_stats)
    df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
    
    # Filter classes with at least 5 samples
    df_filtered = df[df['num_samples'] >= 5].copy()
    
    # Plot - now with 3x2 grid for more metrics
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Accuracy distribution
    axes[0, 0].hist(df_filtered['accuracy'], bins=50, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(df_filtered['accuracy'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df_filtered["accuracy"].mean():.1f}%')
    axes[0, 0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Classes', fontsize=12)
    axes[0, 0].set_title('Per-Class Accuracy Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision distribution
    axes[0, 1].hist(df_filtered['precision'], bins=50, color='green', edgecolor='black')
    axes[0, 1].axvline(df_filtered['precision'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df_filtered["precision"].mean():.1f}%')
    axes[0, 1].set_xlabel('Precision (%)', fontsize=12)
    axes[0, 1].set_ylabel('Number of Classes', fontsize=12)
    axes[0, 1].set_title('Per-Class Precision Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Recall distribution
    axes[1, 0].hist(df_filtered['recall'], bins=50, color='orange', edgecolor='black')
    axes[1, 0].axvline(df_filtered['recall'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df_filtered["recall"].mean():.1f}%')
    axes[1, 0].set_xlabel('Recall (%)', fontsize=12)
    axes[1, 0].set_ylabel('Number of Classes', fontsize=12)
    axes[1, 0].set_title('Per-Class Recall Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. F1-Score distribution
    axes[1, 1].hist(df_filtered['f1_score'], bins=50, color='purple', edgecolor='black')
    axes[1, 1].axvline(df_filtered['f1_score'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df_filtered["f1_score"].mean():.1f}%')
    axes[1, 1].set_xlabel('F1-Score (%)', fontsize=12)
    axes[1, 1].set_ylabel('Number of Classes', fontsize=12)
    axes[1, 1].set_title('Per-Class F1-Score Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Precision vs Recall scatter
    axes[2, 0].scatter(df_filtered['recall'], df_filtered['precision'], 
                       alpha=0.5, s=30, c='steelblue')
    axes[2, 0].plot([0, 100], [0, 100], 'r--', alpha=0.3, label='Perfect')
    axes[2, 0].set_xlabel('Recall (%)', fontsize=12)
    axes[2, 0].set_ylabel('Precision (%)', fontsize=12)
    axes[2, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim([0, 105])
    axes[2, 0].set_ylim([0, 105])
    
    # 6. Metrics vs Sample Size
    axes[2, 1].scatter(df_filtered['num_samples'], df_filtered['f1_score'], 
                       alpha=0.5, s=30, c='purple', label='F1-Score')
    axes[2, 1].scatter(df_filtered['num_samples'], df_filtered['accuracy'], 
                       alpha=0.5, s=30, c='steelblue', label='Accuracy')
    axes[2, 1].set_xlabel('Number of Samples', fontsize=12)
    axes[2, 1].set_ylabel('Score (%)', fontsize=12)
    axes[2, 1].set_title('Metrics vs Sample Size', fontsize=14, fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_analysis.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: per_class_analysis.png")
    print(f"   ✓ Saved: per_class_metrics.csv")
    
    # Print summary
    print(f"\n   Summary:")
    print(f"     Mean accuracy:  {df_filtered['accuracy'].mean():.2f}%")
    print(f"     Mean precision: {df_filtered['precision'].mean():.2f}%")
    print(f"     Mean recall:    {df_filtered['recall'].mean():.2f}%")
    print(f"     Mean F1-score:  {df_filtered['f1_score'].mean():.2f}%")


def plot_confidence_analysis(results, output_dir, config):
    """Plot confidence score analysis"""
    print(f"\n📊 Analyzing confidence scores...")
    
    correct = results['correct']
    pred_conf = results['pred_confidences']
    true_conf = results['true_confidences']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Confidence distribution for correct vs incorrect
    axes[0, 0].hist(pred_conf[correct], bins=50, alpha=0.7, label='Correct', 
                    color='green', edgecolor='black')
    axes[0, 0].hist(pred_conf[~correct], bins=50, alpha=0.7, label='Incorrect', 
                    color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Confidence vs Accuracy
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracy = []
    
    for i in range(len(bins) - 1):
        mask = (pred_conf >= bins[i]) & (pred_conf < bins[i+1])
        if mask.sum() > 0:
            acc = correct[mask].mean() * 100
        else:
            acc = 0
        bin_accuracy.append(acc)
    
    axes[0, 1].plot(bin_centers, bin_accuracy, marker='o', linewidth=2, 
                    markersize=6, color='steelblue')
    axes[0, 1].plot([0, 1], [0, 100], 'r--', alpha=0.5, label='Perfect calibration')
    axes[0, 1].set_xlabel('Confidence', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Model Calibration', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 100])
    
    # 3. True class confidence for correct vs incorrect
    axes[1, 0].hist(true_conf[correct], bins=50, alpha=0.7, label='Correct', 
                    color='green', edgecolor='black')
    axes[1, 0].hist(true_conf[~correct], bins=50, alpha=0.7, label='Incorrect', 
                    color='red', edgecolor='black')
    axes[1, 0].set_xlabel('True Class Confidence', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('True Class Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Confidence statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Confidence Statistics
    {'='*40}
    
    Correct Predictions:
      Mean confidence: {pred_conf[correct].mean():.3f}
      Median confidence: {np.median(pred_conf[correct]):.3f}
      Std confidence: {pred_conf[correct].std():.3f}
    
    Incorrect Predictions:
      Mean confidence: {pred_conf[~correct].mean():.3f}
      Median confidence: {np.median(pred_conf[~correct]):.3f}
      Std confidence: {pred_conf[~correct].std():.3f}
    
    Overall:
      Mean confidence: {pred_conf.mean():.3f}
      Samples with >90% conf: {(pred_conf > 0.9).sum()} ({(pred_conf > 0.9).mean()*100:.1f}%)
      Samples with <50% conf: {(pred_conf < 0.5).sum()} ({(pred_conf < 0.5).mean()*100:.1f}%)
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: confidence_analysis.png")


def plot_topk_accuracy(results, output_dir, config):
    """Plot top-k accuracy curve"""
    max_k = config['topk_max']
    
    print(f"\n📊 Analyzing top-k accuracy...")
    
    labels = results['labels']
    top5_preds = results['top5_predictions']
    probs = results['probabilities']
    
    # Compute top-k accuracy for k=1 to max_k
    topk_accuracies = []
    
    for k in range(1, max_k + 1):
        # Get top-k predictions
        topk_preds = np.argsort(probs, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = np.any(topk_preds == labels.reshape(-1, 1), axis=1)
        accuracy = correct.mean() * 100
        topk_accuracies.append(accuracy)
    
    # Plot
    fig, ax = plt.subplots(figsize=config['figsize_small'])
    
    k_values = list(range(1, max_k + 1))
    ax.plot(k_values, topk_accuracies, marker='o', linewidth=2, 
            markersize=6, color='steelblue')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Top-k Accuracy (%)', fontsize=12)
    ax.set_title('Top-k Accuracy Curve', fontsize=14, fontweight='bold')
    ax.set_xlim([1, max_k])
    ax.set_ylim([0, 100])
    
    # Annotate key points
    ax.annotate(f'Top-1: {topk_accuracies[0]:.2f}%', 
                xy=(1, topk_accuracies[0]), xytext=(3, topk_accuracies[0] - 10),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    
    ax.annotate(f'Top-5: {topk_accuracies[4]:.2f}%', 
                xy=(5, topk_accuracies[4]), xytext=(7, topk_accuracies[4] - 10),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_accuracy.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: topk_accuracy.png")


def plot_error_analysis(results, output_dir, config):
    """Analyze common misclassifications"""
    top_n = config['error_analysis_top_n']
    
    print(f"\n📊 Analyzing common errors...")
    
    labels = results['labels']
    preds = results['predictions']
    
    # Find incorrect predictions
    incorrect_mask = (preds != labels)
    incorrect_labels = labels[incorrect_mask]
    incorrect_preds = preds[incorrect_mask]
    
    # Count confusion pairs
    confusion_pairs = [(true, pred) for true, pred in zip(incorrect_labels, incorrect_preds)]
    confusion_counts = Counter(confusion_pairs)
    
    # Get top N most common errors
    top_errors = confusion_counts.most_common(top_n)
    
    # Create DataFrame
    error_df = pd.DataFrame([
        {
            'true_class': true,
            'predicted_class': pred,
            'count': count,
            'percentage': count / len(incorrect_labels) * 100
        }
        for (true, pred), count in top_errors
    ])
    
    error_df.to_csv(output_dir / 'common_errors.csv', index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=config['figsize_medium'])
    
    labels_str = [f"{row['true_class']}→{row['predicted_class']}" 
                  for _, row in error_df.iterrows()]
    
    bars = ax.barh(range(len(error_df)), error_df['count'].values, color='coral')
    ax.set_yticks(range(len(error_df)))
    ax.set_yticklabels(labels_str)
    ax.set_xlabel('Number of Errors', fontsize=12)
    ax.set_title(f'Top {top_n} Most Common Misclassifications', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add count labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {int(width)}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'common_errors.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: common_errors.png")
    print(f"   ✓ Saved: common_errors.csv")


def compute_macro_micro_metrics(results):
    """Compute macro and micro averaged metrics"""
    labels = results['labels']
    preds = results['predictions']
    num_classes = results['probabilities'].shape[1]
    
    # Per-class TP, FP, FN
    tp_per_class = np.zeros(num_classes)
    fp_per_class = np.zeros(num_classes)
    fn_per_class = np.zeros(num_classes)
    
    for c in range(num_classes):
        tp_per_class[c] = ((preds == c) & (labels == c)).sum()
        fp_per_class[c] = ((preds == c) & (labels != c)).sum()
        fn_per_class[c] = ((preds != c) & (labels == c)).sum()
    
    # Micro-averaged metrics (aggregate over all classes)
    tp_total = tp_per_class.sum()
    fp_total = fp_per_class.sum()
    fn_total = fn_per_class.sum()
    
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Macro-averaged metrics (average of per-class metrics)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    
    for c in range(num_classes):
        if tp_per_class[c] + fp_per_class[c] > 0:
            precision_per_class[c] = tp_per_class[c] / (tp_per_class[c] + fp_per_class[c])
        if tp_per_class[c] + fn_per_class[c] > 0:
            recall_per_class[c] = tp_per_class[c] / (tp_per_class[c] + fn_per_class[c])
        if precision_per_class[c] + recall_per_class[c] > 0:
            f1_per_class[c] = 2 * precision_per_class[c] * recall_per_class[c] / (precision_per_class[c] + recall_per_class[c])
    
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()
    
    # Weighted averages (weighted by class frequency)
    class_counts = np.array([(labels == c).sum() for c in range(num_classes)])
    total_samples = class_counts.sum()
    
    weighted_precision = (precision_per_class * class_counts).sum() / total_samples if total_samples > 0 else 0
    weighted_recall = (recall_per_class * class_counts).sum() / total_samples if total_samples > 0 else 0
    weighted_f1 = (f1_per_class * class_counts).sum() / total_samples if total_samples > 0 else 0
    
    return {
        'micro_precision': micro_precision * 100,
        'micro_recall': micro_recall * 100,
        'micro_f1': micro_f1 * 100,
        'macro_precision': macro_precision * 100,
        'macro_recall': macro_recall * 100,
        'macro_f1': macro_f1 * 100,
        'weighted_precision': weighted_precision * 100,
        'weighted_recall': weighted_recall * 100,
        'weighted_f1': weighted_f1 * 100,
    }


def plot_precision_recall_f1_summary(results, output_dir, config, num_classes=226):
    """Plot detailed precision, recall, F1 summary with best/worst classes"""
    print(f"\n📊 Generating precision/recall/F1 detailed plots...")
    
    labels = results['labels']
    preds = results['predictions']
    
    # Compute per-class metrics
    per_class_stats = []
    
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        num_samples = (labels == c).sum()
        
        if num_samples > 0:
            accuracy = tp / num_samples * 100
        else:
            accuracy = 0.0
        
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0.0
        
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        per_class_stats.append({
            'class': c,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': num_samples
        })
    
    df = pd.DataFrame(per_class_stats)
    df_filtered = df[df['num_samples'] >= 5].copy()
    
    # Create 2x2 figure for best/worst analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top 15 Precision
    top_precision = df_filtered.nlargest(15, 'precision')
    axes[0, 0].barh(range(15), top_precision['precision'].values, color='green', alpha=0.7)
    axes[0, 0].set_yticks(range(15))
    axes[0, 0].set_yticklabels([f"Class {int(c)}" for c in top_precision['class'].values])
    axes[0, 0].set_xlabel('Precision (%)', fontsize=12)
    axes[0, 0].set_title('Top 15 Classes by Precision', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim([0, 105])
    
    # 2. Top 15 Recall
    top_recall = df_filtered.nlargest(15, 'recall')
    axes[0, 1].barh(range(15), top_recall['recall'].values, color='orange', alpha=0.7)
    axes[0, 1].set_yticks(range(15))
    axes[0, 1].set_yticklabels([f"Class {int(c)}" for c in top_recall['class'].values])
    axes[0, 1].set_xlabel('Recall (%)', fontsize=12)
    axes[0, 1].set_title('Top 15 Classes by Recall', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim([0, 105])
    
    # 3. Top 15 F1-Score
    top_f1 = df_filtered.nlargest(15, 'f1_score')
    axes[1, 0].barh(range(15), top_f1['f1_score'].values, color='purple', alpha=0.7)
    axes[1, 0].set_yticks(range(15))
    axes[1, 0].set_yticklabels([f"Class {int(c)}" for c in top_f1['class'].values])
    axes[1, 0].set_xlabel('F1-Score (%)', fontsize=12)
    axes[1, 0].set_title('Top 15 Classes by F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim([0, 105])
    
    # 4. Bottom 15 F1-Score (worst performing)
    bottom_f1 = df_filtered.nsmallest(15, 'f1_score')
    axes[1, 1].barh(range(15), bottom_f1['f1_score'].values, color='red', alpha=0.7)
    axes[1, 1].set_yticks(range(15))
    axes[1, 1].set_yticklabels([f"Class {int(c)}" for c in bottom_f1['class'].values])
    axes[1, 1].set_xlabel('F1-Score (%)', fontsize=12)
    axes[1, 1].set_title('Bottom 15 Classes by F1-Score', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_f1_rankings.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: precision_recall_f1_rankings.png")


def plot_macro_micro_comparison(results, output_dir, config):
    """Plot comparison of macro, micro, and weighted metrics"""
    print(f"\n📊 Computing macro/micro/weighted averages...")
    
    metrics = compute_macro_micro_metrics(results)
    
    # Save to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'aggregate_metrics.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bar chart comparing metrics
    metric_types = ['Micro', 'Macro', 'Weighted']
    precision_vals = [metrics['micro_precision'], metrics['macro_precision'], metrics['weighted_precision']]
    recall_vals = [metrics['micro_recall'], metrics['macro_recall'], metrics['weighted_recall']]
    f1_vals = [metrics['micro_f1'], metrics['macro_f1'], metrics['weighted_f1']]
    
    x = np.arange(len(metric_types))
    width = 0.25
    
    axes[0].bar(x - width, precision_vals, width, label='Precision', color='green', alpha=0.8)
    axes[0].bar(x, recall_vals, width, label='Recall', color='orange', alpha=0.8)
    axes[0].bar(x + width, f1_vals, width, label='F1-Score', color='purple', alpha=0.8)
    
    axes[0].set_xlabel('Averaging Method', fontsize=12)
    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('Aggregate Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_types)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 100])
    
    # 2. Text summary
    axes[1].axis('off')
    summary_text = f"""
    Aggregate Metrics Summary
    {'='*50}
    
    MICRO-AVERAGED (Overall):
      Precision: {metrics['micro_precision']:.2f}%
      Recall:    {metrics['micro_recall']:.2f}%
      F1-Score:  {metrics['micro_f1']:.2f}%
    
    MACRO-AVERAGED (Per-Class):
      Precision: {metrics['macro_precision']:.2f}%
      Recall:    {metrics['macro_recall']:.2f}%
      F1-Score:  {metrics['macro_f1']:.2f}%
    
    WEIGHTED-AVERAGED (By Frequency):
      Precision: {metrics['weighted_precision']:.2f}%
      Recall:    {metrics['weighted_recall']:.2f}%
      F1-Score:  {metrics['weighted_f1']:.2f}%
    
    {'='*50}
    Note:
    - Micro: Aggregate over all instances
    - Macro: Simple average of per-class metrics
    - Weighted: Average weighted by class frequency
    """
    axes[1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_metrics.png', dpi=config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: aggregate_metrics.png")
    print(f"   ✓ Saved: aggregate_metrics.csv")
    
    # Print summary
    print(f"\n   Summary:")
    print(f"     Micro F1:    {metrics['micro_f1']:.2f}%")
    print(f"     Macro F1:    {metrics['macro_f1']:.2f}%")
    print(f"     Weighted F1: {metrics['weighted_f1']:.2f}%")


def generate_summary_report(results, output_dir, model_config):
    """Generate a comprehensive text report"""
    print(f"\n📄 Generating summary report...")
    
    # Compute aggregate metrics
    agg_metrics = compute_macro_micro_metrics(results)
    
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Model info
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model size: {model_config.get('model_size', 'N/A')}\n")
        f.write(f"Number of classes: {model_config.get('num_classes', 'N/A')}\n")
        f.write(f"Dropout: {model_config.get('dropout', 'N/A')}\n")
        f.write(f"Batch size: {model_config.get('batch_size', 'N/A')}\n")
        f.write(f"Learning rate: {model_config.get('lr', 'N/A')}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%\n")
        f.write(f"Total samples: {len(results['labels'])}\n")
        f.write(f"Correct predictions: {results['correct'].sum()}\n")
        f.write(f"Incorrect predictions: {(~results['correct']).sum()}\n")
        f.write("\n")
        
        # Aggregate metrics
        f.write("AGGREGATE METRICS\n")
        f.write("-"*80 + "\n")
        f.write("Micro-Averaged (Overall):\n")
        f.write(f"  Precision: {agg_metrics['micro_precision']:.2f}%\n")
        f.write(f"  Recall:    {agg_metrics['micro_recall']:.2f}%\n")
        f.write(f"  F1-Score:  {agg_metrics['micro_f1']:.2f}%\n")
        f.write("\n")
        f.write("Macro-Averaged (Per-Class):\n")
        f.write(f"  Precision: {agg_metrics['macro_precision']:.2f}%\n")
        f.write(f"  Recall:    {agg_metrics['macro_recall']:.2f}%\n")
        f.write(f"  F1-Score:  {agg_metrics['macro_f1']:.2f}%\n")
        f.write("\n")
        f.write("Weighted-Averaged (By Frequency):\n")
        f.write(f"  Precision: {agg_metrics['weighted_precision']:.2f}%\n")
        f.write(f"  Recall:    {agg_metrics['weighted_recall']:.2f}%\n")
        f.write(f"  F1-Score:  {agg_metrics['weighted_f1']:.2f}%\n")
        f.write("\n")
        
        # Confidence stats
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-"*80 + "\n")
        correct = results['correct']
        pred_conf = results['pred_confidences']
        f.write(f"Mean confidence (correct): {pred_conf[correct].mean():.3f}\n")
        f.write(f"Mean confidence (incorrect): {pred_conf[~correct].mean():.3f}\n")
        f.write(f"Overall mean confidence: {pred_conf.mean():.3f}\n")
        f.write(f"High confidence (>90%): {(pred_conf > 0.9).sum()} samples\n")
        f.write(f"Low confidence (<50%): {(pred_conf < 0.5).sum()} samples\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"   ✓ Saved: evaluation_report.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = CONFIG
    
    # Determine paths
    if config['checkpoint_path']:
        checkpoint_path = Path(config['checkpoint_path'])
        output_dir = checkpoint_path.parent
    elif config['exp_name']:
        checkpoint_path = Path('checkpoints') / config['exp_name'] / 'best_model.pth'
        output_dir = Path('checkpoints') / config['exp_name']
    else:
        # Find most recent checkpoint
        checkpoint_dir = Path('checkpoints')
        if not checkpoint_dir.exists():
            print("❌ No checkpoints directory found!")
            print("   Please set 'exp_name' or 'checkpoint_path' in CONFIG")
            return
        
        subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("❌ No experiment directories found!")
            return
        
        # Get most recent
        latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
        checkpoint_path = latest_dir / 'best_model.pth'
        output_dir = latest_dir
        print(f"📁 Auto-detected most recent experiment: {latest_dir.name}")
    
    if config['output_dir']:
        output_dir = Path(config['output_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION & VISUALIZATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print("="*80)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load config
    config_path = output_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print("\n✓ Loaded model configuration")
    else:
        model_config = {
            'num_classes': 226,
            'model_size': 'base',
            'dropout': 0.1
        }
        print("\n⚠ Config not found, using defaults")
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\n🏗️  Loading model...")
    model = create_model(
        num_classes=model_config.get('num_classes', 226),
        model_size=model_config.get('model_size', 'base'),
        dropout=model_config.get('dropout', 0.1)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Create dataloader
    print("\n📦 Loading test data...")
    test_loader = create_dataloader(
        config['data_root'], 'test',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=False
    )
    print(f"✓ Test loader created: {len(test_loader)} batches")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, model_config.get('num_classes', 226))
    
    # Generate all plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_confusion_matrix(results, output_dir, config)
    plot_per_class_accuracy(results, output_dir, config, model_config.get('num_classes', 226))
    plot_precision_recall_f1_summary(results, output_dir, config, model_config.get('num_classes', 226))
    plot_confidence_analysis(results, output_dir, config)
    plot_topk_accuracy(results, output_dir, config)
    plot_error_analysis(results, output_dir, config)
    plot_macro_micro_comparison(results, output_dir, config)
    generate_summary_report(results, output_dir, model_config)
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS AND ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - per_class_analysis.png")
    print("  - per_class_metrics.csv")
    print("  - precision_recall_f1_rankings.png")
    print("  - confidence_analysis.png")
    print("  - topk_accuracy.png")
    print("  - common_errors.png")
    print("  - common_errors.csv")
    print("  - aggregate_metrics.png")
    print("  - aggregate_metrics.csv")
    print("  - evaluation_report.txt")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()