"""
analyze_raw_autsl.py - Publication-Ready Analysis (IEEE Format Tuned)
Optimized for strictly 7.16" width (IEEE \textwidth) to ensure 100% font size accuracy.
"""

import os
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
#                     CONFIGURATION
# ==============================================================================

DATASET_PATH = 'autsl'
TRAIN_LABELS = os.path.join(DATASET_PATH, 'train_labels.csv')
VAL_LABELS = os.path.join(DATASET_PATH, 'val_labels.csv')
TEST_LABELS = os.path.join(DATASET_PATH, 'test_labels.csv')
OUTPUT_DIR = 'paper_figures'
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']

# --- IEEE PRECISE STYLING ---
# IEEE allows 8pt as minimum body text in figures.
sns.set_theme(style="white", context="paper", font="serif")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,              # Base font size for IEEE
    'axes.labelsize': 8,
    'axes.titlesize': 9,         # Slightly larger for subplot titles
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'axes.linewidth': 0.8,       # Thinner lines for smaller figure
    'lines.linewidth': 1.2,      # Thinner plot lines
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#DDDDDD',
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
    'savefig.dpi': 600,          # High res for print
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# Professional Palette (Mako)
PALETTE = sns.color_palette("mako", n_colors=6)
COLOR_MAIN = PALETTE[1]
COLOR_SEC = PALETTE[3]
COLOR_ACCENT = '#C44E52'
COLOR_NEUTRAL = '#555555'

# IEEE Standard Widths (Inches)
FIG_WIDTH_DOUBLE = 7.16  # Fits \textwidth perfectly
FIG_HEIGHT_STD = 2.8     # Compact height for papers

# ==============================================================================
#                     DATA EXTRACTION
# ==============================================================================

def create_output_directory():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}/")

def load_labels():
    print("\n📂 Loading dataset labels...")
    try:
        train_df = pd.read_csv(TRAIN_LABELS, header=None, names=['filename', 'label'])
        val_df = pd.read_csv(VAL_LABELS, header=None, names=['filename', 'label'])
        test_df = pd.read_csv(TEST_LABELS, header=None, names=['filename', 'label'])
        
        for df, split in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
            df['signer'] = df['filename'].str.split('_').str[0]
            df['split'] = split
        
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return combined_df, train_df, val_df, test_df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None, None

def find_video_file(dataset_path, split, filename):
    for ext in VIDEO_EXTENSIONS:
        video_path = Path(dataset_path) / split / f"{filename}_color{ext}"
        if video_path.exists(): return video_path
    for ext in VIDEO_EXTENSIONS:
        video_path = Path(dataset_path) / split / f"{filename}{ext}"
        if video_path.exists(): return video_path
    return None

def extract_video_info(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        info = {
            'length': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
        }
        cap.release()
        return info
    except:
        return None

def analyze_videos(combined_df, sample_size=None):
    print("\n📹 Analyzing videos...")
    video_stats, missing = [], 0
    df_to_analyze = combined_df.sample(n=min(sample_size or len(combined_df), len(combined_df)), random_state=42) if sample_size else combined_df
    
    for idx, row in tqdm(df_to_analyze.iterrows(), total=len(df_to_analyze), desc="  Analyzing"):
        video_path = find_video_file(DATASET_PATH, row['split'], row['filename'])
        if not video_path:
            missing += 1
            continue
        info = extract_video_info(video_path)
        if info:
            info.update({'label': row['label'], 'signer': row['signer'], 'split': row['split']})
            video_stats.append(info)
            
    if missing: print(f"   ⚠️  {missing} videos not found")
    return pd.DataFrame(video_stats) if video_stats else None

# ==============================================================================
#                     FIGURES (IEEE TUNED)
# ==============================================================================

def add_stats_box(ax, text):
    """Adds a compact stats box suitable for small figures."""
    ax.text(0.98, 0.95, text, transform=ax.transAxes, 
            fontsize=6.5, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#CCCCCC', linewidth=0.5))

def plot_class_distribution(df):
    print("\n📊 Figure A: Class Distribution...")
    class_counts = df.groupby('label').size().values
    
    # EXACT IEEE DIMENSIONS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_HEIGHT_STD))
    
    # LEFT: Violin
    violin_data = pd.DataFrame({'Counts': class_counts, 'Category': 'Dist'})
    sns.violinplot(y='Counts', x='Category', data=violin_data, ax=ax1, 
                   color=COLOR_MAIN, inner=None, linewidth=0.8, alpha=0.3)
    sns.stripplot(y='Counts', x='Category', data=violin_data, ax=ax1, 
                  color=COLOR_NEUTRAL, size=2, alpha=0.4, jitter=0.25, zorder=2)
    
    ax1.set_ylabel('Samples / Class')
    ax1.set_xlabel('')
    ax1.set_title('(a) Class Imbalance')
    ax1.yaxis.grid(True, linestyle=':', alpha=0.6)

    q1, q3 = np.percentile(class_counts, [25, 75])
    add_stats_box(ax1, f"N={len(class_counts)}\nMean={class_counts.mean():.0f}\nIQR={q3-q1:.0f}")

    # RIGHT: KDE
    sns.histplot(class_counts, kde=True, ax=ax2, color=COLOR_SEC, 
                 stat="density", linewidth=0, alpha=0.3, binwidth=5)
    
    lines = ax2.get_lines()
    if lines: 
        lines[0].set_color(COLOR_MAIN)
        lines[0].set_linewidth(1.5)

    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Density Est.')
    ax2.yaxis.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{OUTPUT_DIR}/fig_a_class_distribution.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_a_class_distribution.png')
    plt.close()

def plot_signer_distribution(df):
    print("\n📊 Figure B: Signer Distribution...")
    signer_counts = df.groupby('signer').size().sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_HEIGHT_STD))
    
    # LEFT: Gradient Bars
    y_pos = np.arange(len(signer_counts))
    norm = plt.Normalize(signer_counts.min(), signer_counts.max())
    colors = plt.cm.viridis(norm(signer_counts.values))
    
    ax1.barh(y_pos, signer_counts.values, color=colors, height=0.75, edgecolor='none')
    
    # Cleaner Y-axis labels
    short_labels = [f"S{s.split('_')[-1]}" if '_' in s else s for s in signer_counts.index]
    # Only show every 2nd or 3rd label if too many signers to fit in 3 inches height
    if len(short_labels) > 20:
        ax1.set_yticks(y_pos[::2])
        ax1.set_yticklabels(short_labels[::2], fontsize=6)
    else:
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(short_labels, fontsize=6)
        
    ax1.invert_yaxis()
    ax1.set_xlabel('Samples')
    ax1.set_title('(a) Signer Contrib.')
    
    # RIGHT: Boxplot
    sns.boxplot(y=signer_counts.values, ax=ax2, width=0.4, color=COLOR_SEC, fliersize=2)
    sns.stripplot(y=signer_counts.values, ax=ax2, color=COLOR_NEUTRAL, alpha=0.5, jitter=0.2, size=2)
    
    ax2.set_ylabel('Samples / Signer')
    ax2.set_title('(b) Variability')
    ax2.set_xticks([])

    gini = (2 * np.sum(np.arange(1, len(signer_counts)+1) * np.sort(signer_counts.values))) / (len(signer_counts) * signer_counts.sum()) - (len(signer_counts)+1)/len(signer_counts)
    add_stats_box(ax2, f"Signers={len(signer_counts)}\nGini={gini:.2f}")

    plt.tight_layout(pad=0.5)
    plt.savefig(f'{OUTPUT_DIR}/fig_b_signer_distribution.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_b_signer_distribution.png')
    plt.close()

def plot_sequence_lengths(stats_df):
    if stats_df is None: return
    print("\n📊 Figure C: Sequence Lengths...")
    
    lengths = stats_df['length'].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_HEIGHT_STD))
    
    # LEFT: Hist
    sns.histplot(lengths, kde=True, ax=ax1, color=COLOR_MAIN, 
                 element="step", fill=True, alpha=0.2, linewidth=0)
    
    mean_val = np.mean(lengths)
    ax1.axvline(mean_val, color=COLOR_ACCENT, linestyle='--', linewidth=1, label=f'Mean:{mean_val:.0f}')
    
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Length Dist.')
    ax1.legend(loc='upper right', frameon=False, fontsize=6)

    # RIGHT: CDF
    sorted_data = np.sort(lengths)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    
    ax2.plot(sorted_data, yvals, color=COLOR_MAIN, linewidth=1.5)
    
    for p in [0.5, 0.95]:
        val = np.percentile(lengths, p*100)
        ax2.plot([val, val], [0, p], color=COLOR_NEUTRAL, linestyle=':', linewidth=0.8)
        ax2.plot([0, val], [p, p], color=COLOR_NEUTRAL, linestyle=':', linewidth=0.8)
        ax2.text(val+2, p-0.1, f'{val:.0f}', fontsize=7, color=COLOR_NEUTRAL)

    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Cumulative Prob.')
    ax2.set_title('(b) CDF')
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{OUTPUT_DIR}/fig_c_sequence_lengths.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_c_sequence_lengths.png')
    plt.close()

def plot_video_quality(stats_df):
    if stats_df is None: return
    print("\n📊 Figure D: Video Quality...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_HEIGHT_STD))
    
    # LEFT: Resolution
    stats_df['res_str'] = stats_df['width'].astype(str) + '×' + stats_df['height'].astype(str)
    res_counts = stats_df['res_str'].value_counts()
    res_counts = res_counts[~res_counts.index.str.contains('0×0')]
    
    x_pos = np.arange(len(res_counts))
    ax1.bar(x_pos, res_counts.values, color=PALETTE, width=0.5, alpha=0.9)
    
    for i, v in enumerate(res_counts.values):
        ax1.text(i, v + (v*0.05), f"{v:,}", ha='center', va='bottom', fontsize=7)
        
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(res_counts.index, fontsize=7)
    ax1.set_title('(a) Resolution')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, res_counts.max() * 1.3)
    
    # RIGHT: FPS
    fps_counts = stats_df['fps'].round(1).value_counts()
    ax2.hlines(y=fps_counts.index.astype(str), xmin=0, xmax=fps_counts.values, color=COLOR_SEC, alpha=0.6, linewidth=2)
    ax2.plot(fps_counts.values, fps_counts.index.astype(str), "o", markersize=6, color=COLOR_SEC)

    for i, v in enumerate(fps_counts.values):
        ax2.text(v + (v*0.02), i, f" {v:,}", va='center', fontsize=7)

    ax2.set_xlabel('Count')
    ax2.set_ylabel('FPS')
    ax2.set_title('(b) Frame Rate')
    ax2.set_xlim(0, fps_counts.max() * 1.3)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{OUTPUT_DIR}/fig_d_video_quality.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_d_video_quality.png')
    plt.close()

# ==============================================================================
#                     MAIN
# ==============================================================================

def main():
    print("="*60)
    print(" IEEE ANALYSIS (7.16\" Width)")
    print("="*60)
    
    create_output_directory()
    combined_df, train_df, val_df, test_df = load_labels()
    if combined_df is None: return
    
    choice = input("\n[1] All Videos  [2] Sample 2000 (Default): ").strip() or "2"
    stats_df = analyze_videos(combined_df, sample_size=None if choice=="1" else 2000)
    
    plot_class_distribution(combined_df)
    plot_signer_distribution(combined_df)
    plot_sequence_lengths(stats_df)
    plot_video_quality(stats_df)
    
    print("\n✅ Done. Use \\begin{figure*} ... \\end{figure*} in LaTeX.")

if __name__ == '__main__':
    main()