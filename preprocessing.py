"""
AUTSL Preprocessing Verification Script
Checks integrity, statistics, and visualizes processed data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'processed_root': 'processed',  # Path to processed data
    'output_folder': 'verification_results',  # Folder name for outputs
    'splits': ['train', 'val', 'test'],
    'visualize_samples': 5,  # Number of random samples to visualize
    'check_nan': True,
    'check_shapes': True,
    'generate_plots': True,
}

# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def load_metadata(processed_root, split):
    """Load metadata file for a split"""
    metadata_file = Path(processed_root) / f"{split}_metadata.csv"
    if not metadata_file.exists():
        return None
    return pd.read_csv(metadata_file)


def verify_file_integrity(processed_root, metadata_df, split):
    """Check if all files exist and are loadable"""
    print(f"\n{'='*60}")
    print(f"Verifying {split.upper()} split file integrity...")
    print(f"{'='*60}")
    
    processed_root = Path(processed_root)
    issues = []
    valid_samples = []
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Checking files"):
        file_path = processed_root / row['file_path']
        
        # Check if file exists
        if not file_path.exists():
            issues.append(f"Missing file: {file_path}")
            continue
        
        # Try to load
        try:
            data = np.load(file_path)
            
            # Check for NaN
            if CONFIG['check_nan'] and np.isnan(data).any():
                issues.append(f"NaN values in: {file_path}")
            
            # Check shape
            if CONFIG['check_shapes']:
                expected_keypoints = row.get('num_keypoints', 56)
                if data.shape[1] != expected_keypoints:
                    issues.append(f"Wrong shape in {file_path}: {data.shape}, expected (*, {expected_keypoints}, 3)")
                
                if data.shape[2] != 3:
                    issues.append(f"Wrong dimension in {file_path}: {data.shape}, expected (*, *, 3)")
            
            valid_samples.append({
                'path': file_path,
                'shape': data.shape,
                'label': row['label'],
                'data': data
            })
            
        except Exception as e:
            issues.append(f"Failed to load {file_path}: {str(e)}")
    
    print(f"\n✓ Valid samples: {len(valid_samples)}/{len(metadata_df)}")
    if issues:
        print(f"✗ Issues found: {len(issues)}")
        print("\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("✓ No issues found!")
    
    return valid_samples, issues


def compute_statistics(valid_samples, split):
    """Compute and display statistics"""
    print(f"\n{'='*60}")
    print(f"Statistics for {split.upper()} split")
    print(f"{'='*60}")
    
    if not valid_samples:
        print("No valid samples to analyze")
        return None
    
    # Extract info
    shapes = [s['shape'] for s in valid_samples]
    labels = [s['label'] for s in valid_samples]
    
    # Temporal statistics
    temporal_lengths = [s[0] for s in shapes]
    num_keypoints = [s[1] for s in shapes]
    
    stats = {
        'num_samples': len(valid_samples),
        'num_classes': len(set(labels)),
        'temporal_length': {
            'mean': np.mean(temporal_lengths),
            'std': np.std(temporal_lengths),
            'min': np.min(temporal_lengths),
            'max': np.max(temporal_lengths),
            'median': np.median(temporal_lengths)
        },
        'num_keypoints': {
            'unique': list(set(num_keypoints)),
            'mode': Counter(num_keypoints).most_common(1)[0][0]
        },
        'label_distribution': Counter(labels)
    }
    
    # Print statistics
    print(f"\nDataset Overview:")
    print(f"  Samples: {stats['num_samples']}")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Keypoints: {stats['num_keypoints']['mode']}")
    
    print(f"\nTemporal Statistics:")
    print(f"  Mean length: {stats['temporal_length']['mean']:.2f} frames")
    print(f"  Std: {stats['temporal_length']['std']:.2f}")
    print(f"  Min: {stats['temporal_length']['min']} frames")
    print(f"  Max: {stats['temporal_length']['max']} frames")
    print(f"  Median: {stats['temporal_length']['median']:.1f} frames")
    
    print(f"\nLabel Distribution:")
    most_common = stats['label_distribution'].most_common(5)
    print(f"  Most common classes:")
    for label, count in most_common:
        print(f"    Class {label}: {count} samples ({100*count/stats['num_samples']:.1f}%)")
    
    least_common = stats['label_distribution'].most_common()[-5:]
    print(f"  Least common classes:")
    for label, count in least_common:
        print(f"    Class {label}: {count} samples ({100*count/stats['num_samples']:.1f}%)")
    
    # Check for class imbalance
    counts = list(stats['label_distribution'].values())
    imbalance_ratio = max(counts) / min(counts)
    print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 10:
        print(f"  ⚠ Warning: Significant class imbalance detected!")
    
    return stats


def check_data_quality(valid_samples, split):
    """Check for data quality issues"""
    print(f"\n{'='*60}")
    print(f"Data Quality Checks for {split.upper()}")
    print(f"{'='*60}")
    
    issues = []
    
    # Sample random videos for detailed check
    num_check = min(100, len(valid_samples))
    check_samples = np.random.choice(valid_samples, num_check, replace=False)
    
    zero_sequences = 0
    low_confidence = 0
    static_sequences = 0
    
    for sample in tqdm(check_samples, desc="Quality checks"):
        data = sample['data']  # (T, J, 3)
        
        # Check for all-zero sequences
        if np.all(data == 0):
            zero_sequences += 1
            continue
        
        # Check confidence/visibility values
        confidence = data[:, :, 2]  # (T, J)
        mean_confidence = np.mean(confidence[confidence > 0])
        if mean_confidence < 0.3:
            low_confidence += 1
        
        # Check for static sequences (no movement)
        coords = data[:, :, :2]  # (T, J, 2)
        movement = np.std(coords, axis=0).mean()
        if movement < 0.01:
            static_sequences += 1
    
    print(f"\nQuality Issues (checked {num_check} samples):")
    print(f"  All-zero sequences: {zero_sequences} ({100*zero_sequences/num_check:.1f}%)")
    print(f"  Low confidence: {low_confidence} ({100*low_confidence/num_check:.1f}%)")
    print(f"  Static sequences: {static_sequences} ({100*static_sequences/num_check:.1f}%)")
    
    if zero_sequences > num_check * 0.05:
        print(f"  ⚠ Warning: >5% zero sequences detected!")
    
    return {
        'zero_sequences': zero_sequences,
        'low_confidence': low_confidence,
        'static_sequences': static_sequences,
        'checked_samples': num_check
    }


def visualize_samples(valid_samples, split, num_samples=5):
    """Visualize random samples"""
    if not CONFIG['generate_plots']:
        return
    
    print(f"\n{'='*60}")
    print(f"Generating visualizations for {split.upper()}...")
    print(f"{'='*60}")
    
    # Sample random videos
    num_samples = min(num_samples, len(valid_samples))
    samples = np.random.choice(valid_samples, num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples):
        data = sample['data']  # (T, J, 3)
        label = sample['label']
        
        # Plot 1: Temporal trajectory (X coordinates over time)
        ax1 = axes[idx, 0]
        for j in range(data.shape[1]):
            ax1.plot(data[:, j, 0], alpha=0.3, linewidth=0.5)
        ax1.set_title(f"Sample {idx+1} - X Trajectories (Label: {label})")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("X Coordinate")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Y coordinates over time
        ax2 = axes[idx, 1]
        for j in range(data.shape[1]):
            ax2.plot(data[:, j, 1], alpha=0.3, linewidth=0.5)
        ax2.set_title(f"Y Trajectories")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Y Coordinate")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence/Visibility over time
        ax3 = axes[idx, 2]
        confidence = data[:, :, 2]
        mean_conf = confidence.mean(axis=1)
        ax3.plot(mean_conf, color='blue', linewidth=2)
        ax3.fill_between(range(len(mean_conf)), 
                         confidence.min(axis=1), 
                         confidence.max(axis=1), 
                         alpha=0.3)
        ax3.set_title(f"Confidence/Visibility")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Confidence")
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(CONFIG['processed_root']) / f"{split}_sample_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_distribution_summary(all_stats):
    """Plot summary statistics across splits"""
    if not CONFIG['generate_plots'] or not all_stats:
        return
    
    print(f"\n{'='*60}")
    print(f"Generating summary plots...")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of samples per split
    ax1 = axes[0, 0]
    splits = list(all_stats.keys())
    counts = [all_stats[s]['num_samples'] for s in splits]
    ax1.bar(splits, counts, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_title('Samples per Split', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    for i, (split, count) in enumerate(zip(splits, counts)):
        ax1.text(i, count + max(counts)*0.02, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Temporal length distribution
    ax2 = axes[0, 1]
    for split in splits:
        stats = all_stats[split]
        temp_stats = stats['temporal_length']
        ax2.bar(split, temp_stats['mean'], alpha=0.7, label=f"{split}")
        ax2.errorbar(split, temp_stats['mean'], yerr=temp_stats['std'], 
                    fmt='none', color='black', capsize=5)
    ax2.set_title('Average Temporal Length', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frames (mean ± std)')
    ax2.legend()
    
    # Plot 3: Class distribution (for train split)
    if 'train' in all_stats:
        ax3 = axes[1, 0]
        label_dist = all_stats['train']['label_distribution']
        top_classes = dict(Counter(label_dist).most_common(20))
        ax3.bar(range(len(top_classes)), list(top_classes.values()), color='#3498db')
        ax3.set_title('Top 20 Classes (Train)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Class Index')
        ax3.set_ylabel('Number of Samples')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    headers = ['Split', 'Samples', 'Classes', 'Avg Length', 'Keypoints']
    
    for split in splits:
        stats = all_stats[split]
        row = [
            split.upper(),
            stats['num_samples'],
            stats['num_classes'],
            f"{stats['temporal_length']['mean']:.1f}",
            stats['num_keypoints']['mode']
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=['#f0f0f0']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = Path(CONFIG['processed_root']) / 'preprocessing_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def generate_report(all_stats, all_quality, output_path):
    """Generate a text report"""
    print(f"\n{'='*60}")
    print(f"Generating verification report...")
    print(f"{'='*60}")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AUTSL PREPROCESSING VERIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        total_samples = sum(s['num_samples'] for s in all_stats.values())
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Splits: {', '.join(all_stats.keys())}\n\n")
        
        # Per-split details
        for split, stats in all_stats.items():
            f.write(f"\n{'-'*80}\n")
            f.write(f"{split.upper()} SPLIT\n")
            f.write(f"{'-'*80}\n\n")
            
            f.write(f"Samples: {stats['num_samples']}\n")
            f.write(f"Classes: {stats['num_classes']}\n")
            f.write(f"Keypoints: {stats['num_keypoints']['mode']}\n\n")
            
            f.write(f"Temporal Statistics:\n")
            temp = stats['temporal_length']
            f.write(f"  Mean: {temp['mean']:.2f} ± {temp['std']:.2f} frames\n")
            f.write(f"  Range: [{temp['min']}, {temp['max']}] frames\n")
            f.write(f"  Median: {temp['median']:.1f} frames\n\n")
            
            if split in all_quality:
                quality = all_quality[split]
                f.write(f"Quality Metrics (checked {quality['checked_samples']} samples):\n")
                f.write(f"  Zero sequences: {quality['zero_sequences']} "
                       f"({100*quality['zero_sequences']/quality['checked_samples']:.1f}%)\n")
                f.write(f"  Low confidence: {quality['low_confidence']} "
                       f"({100*quality['low_confidence']/quality['checked_samples']:.1f}%)\n")
                f.write(f"  Static sequences: {quality['static_sequences']} "
                       f"({100*quality['static_sequences']/quality['checked_samples']:.1f}%)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN VERIFICATION PIPELINE
# ============================================================================

def main():
    """Run full verification pipeline"""
    print("="*80)
    print("AUTSL PREPROCESSING VERIFICATION")
    print("="*80)
    print(f"Processed data root: {CONFIG['processed_root']}")
    print(f"Splits to verify: {CONFIG['splits']}")
    print("="*80)
    
    processed_root = Path(CONFIG['processed_root'])
    
    # Check if directory exists
    if not processed_root.exists():
        print(f"\nERROR: Processed data directory not found: {processed_root}")
        return
    
    # Check for config file
    config_file = processed_root / 'preprocess_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            preprocess_config = json.load(f)
        print(f"\nPreprocessing configuration:")
        for key, value in preprocess_config.items():
            print(f"  {key}: {value}")
    
    all_stats = {}
    all_quality = {}
    
    # Verify each split
    for split in CONFIG['splits']:
        metadata = load_metadata(processed_root, split)
        
        if metadata is None:
            print(f"\nWarning: No metadata found for {split} split")
            continue
        
        # Run verification
        valid_samples, issues = verify_file_integrity(processed_root, metadata, split)
        stats = compute_statistics(valid_samples, split)
        quality = check_data_quality(valid_samples, split)
        
        if CONFIG['generate_plots']:
            visualize_samples(valid_samples, split, CONFIG['visualize_samples'])
        
        all_stats[split] = stats
        all_quality[split] = quality
    
    # Generate summary
    if all_stats:
        plot_distribution_summary(all_stats)
        report_path = processed_root / 'verification_report.txt'
        generate_report(all_stats, all_quality, report_path)
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {processed_root}")
    if CONFIG['generate_plots']:
        print("  - Sample visualizations: *_sample_visualization.png")
        print("  - Summary plots: preprocessing_summary.png")
    print("  - Verification report: verification_report.txt")


if __name__ == '__main__':
    main()