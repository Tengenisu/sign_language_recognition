"""
compare_models_ieee.py - IEEE-Ready Model Comparison (tiny/small/base)

Complete comparison suite with publication-ready outputs:
- Efficiency metrics (params, FLOPs, latency, throughput, memory)
- Performance metrics (Top-1/5, F1, ECE)
- Pareto plots, calibration, confusion differences
- LaTeX tables, statistical tests

Usage:
    python compare_models_ieee.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from model import create_model
from dataset import create_dataloader

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'models': {
        'tiny': {'checkpoint': 'checkpoints/tiny_optimized_20251202_111336/best_model.pth', 'color': '#1f77b4', 'marker': 'o'},
        'small': {'checkpoint': 'checkpoints/small_optimized_20251202_142636/best_model.pth', 'color': '#ff7f0e', 'marker': 's'},
        'base': {'checkpoint': 'checkpoints/base_optimized_20251202_214508/best_model.pth', 'color': '#2ca02c', 'marker': '^'},
    },
    'data_root': 'processed_autsl',
    'batch_size': 64,
    'num_workers': 4,
    'output_dir': 'model_comparison',
    'latency_warmup': 50,
    'latency_iters': 200,
}

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================================
# EFFICIENCY METRICS
# ============================================================================

def get_input_shape(data_root):
    """Get correct input shape from dataset"""
    try:
        test_loader = create_dataloader(data_root, 'test', batch_size=1, num_workers=0, augment=False)
        for seq, jt, _ in test_loader:
            # seq shape: (B, T, J, features_per_joint)
            return seq.shape[1:4]  # (T, J, features_per_joint)
    except:
        # Default fallback: 64 frames, 181 joints, 3 features (x,y,z)
        return (64, 181, 3)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def measure_latency(model, device, input_shape, warmup=50, iters=200):
    """Measure latency with correct input shape"""
    model.eval()
    T, J, F = input_shape
    seq = torch.randn(1, T, J, F).to(device)
    jt = torch.zeros(1, J, dtype=torch.long).to(device)  # Shape: (1, J)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(seq, jt)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.time()
            _ = model(seq, jt)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    
    return np.median(times), np.percentile(times, 95)

def measure_throughput(model, device, input_shape, batch_size=32):
    """Measure throughput with correct input shape"""
    model.eval()
    T, J, F = input_shape
    seq = torch.randn(batch_size, T, J, F).to(device)
    jt = torch.zeros(batch_size, J, dtype=torch.long).to(device)  # Shape: (B, J)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(seq, jt)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_samples = 0
    start = time.time()
    
    with torch.no_grad():
        for _ in range(50):
            _ = model(seq, jt)
            total_samples += batch_size
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return total_samples / (time.time() - start)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, num_classes=226):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for seq, lab, jt in tqdm(dataloader, desc="  Eval", leave=False):
            seq, lab = seq.to(device), lab.to(device)
            batch_size = seq.size(0)
            jt = jt.unsqueeze(0).expand(batch_size, -1).to(device)
            out = model(seq, jt)
            prob = torch.softmax(out, dim=1)
            _, pred = out.max(1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    correct = (all_preds == all_labels)
    top1 = correct.mean() * 100
    
    # Top-k
    topk = []
    for k in range(1, 21):
        topk_preds = np.argsort(all_probs, axis=1)[:, -k:]
        topk_correct = np.any(topk_preds == all_labels.reshape(-1, 1), axis=1)
        topk.append(topk_correct.mean() * 100)
    
    # Per-class F1
    f1_list = []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_list.append(f1 * 100)
    
    # ECE
    bins = np.linspace(0, 1, 11)
    pred_confs = all_probs[np.arange(len(all_preds)), all_preds]
    ece = 0
    for i in range(len(bins) - 1):
        mask = (pred_confs >= bins[i]) & (pred_confs < bins[i+1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = pred_confs[mask].mean()
            ece += mask.sum() / len(pred_confs) * abs(acc - conf)
    ece *= 100
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'top1': top1,
        'top5': topk[4],
        'topk': topk,
        'per_class_f1': f1_list,
        'macro_f1': np.mean(f1_list),
        'ece': ece,
    }

# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_model(name, ckpt, device, config):
    print(f"\n{'='*60}\nBENCHMARKING: {name.upper()}\n{'='*60}")
    
    # Get input shape from dataset
    print("📐 Detecting input shape...")
    input_shape = get_input_shape(config['data_root'])
    print(f"   Input shape: (T={input_shape[0]}, J={input_shape[1]}, F={input_shape[2]})")
    
    # Load
    ckpt_dir = Path(ckpt).parent
    cfg_path = ckpt_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            mcfg = json.load(f)
    else:
        mcfg = {'num_classes': 226, 'model_size': name, 'dropout': 0.1}
    
    model = create_model(
        num_classes=mcfg.get('num_classes', 226),
        model_size=mcfg.get('model_size', name),
        dropout=mcfg.get('dropout', 0.1)
    )
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    res = {'model': name}
    
    # Params
    res['params_M'] = count_parameters(model)
    res['size_MB'] = Path(ckpt).stat().st_size / (1024 ** 2)
    print(f"Params: {res['params_M']:.2f}M | Size: {res['size_MB']:.2f}MB")
    
    # Latency
    lat_med, lat_p95 = measure_latency(model, device, input_shape, config['latency_warmup'], config['latency_iters'])
    res['latency_ms'] = lat_med
    res['latency_p95'] = lat_p95
    print(f"Latency: {lat_med:.2f}ms (p95: {lat_p95:.2f}ms)")
    
    # Throughput
    tput = measure_throughput(model, device, input_shape, 32)
    res['throughput'] = tput
    print(f"Throughput: {tput:.1f} samples/sec")
    
    # Accuracy
    test_loader = create_dataloader(config['data_root'], 'test', config['batch_size'], config['num_workers'], False)
    eval_res = evaluate_model(model, test_loader, device, mcfg.get('num_classes', 226))
    res.update(eval_res)
    print(f"Top-1: {eval_res['top1']:.2f}% | Top-5: {eval_res['top5']:.2f}% | Macro F1: {eval_res['macro_f1']:.2f}%")
    
    return res

# ============================================================================
# PLOTS
# ============================================================================

def plot_pareto(all_res, out_dir):
    print("\n📊 Pareto (accuracy vs latency)...")
    
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(4.0, 3.2)) # Slightly wider to breathe
    
    # 2. Plot Points
    max_latency = 0
    max_acc = 0
    
    for name, res in all_res.items():
        lat = res['latency_ms']
        acc = res['top1']
        params = res['params_M']
        
        # Track max values for axis limits
        max_latency = max(max_latency, lat)
        max_acc = max(max_acc, acc)

        # FIX: Base size of 30 + scaling ensures small models are visible
        marker_size = 30 + (params * 80)
        
        # Get style from config (assuming CONFIG is global or passed)
        # Using .get() for safety if specific keys missing
        color = CONFIG['models'][name].get('color', 'blue')
        marker = CONFIG['models'][name].get('marker', 'o')
        
        ax.scatter(lat, acc, s=marker_size,
                   color=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidth=0.8, zorder=3,
                   label=f"{name.capitalize()} ({params:.1f}M)")
        
        # FIX: Smart annotation offset
        # If point is high up, annotate below it to prevent clipping
        vertical_offset = -15 if acc > 85 else 8
        
        ax.annotate(name.capitalize(), (lat, acc),
                    xytext=(0, vertical_offset), textcoords='offset points', 
                    fontsize=9, fontweight='bold', ha='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # 3. Add 30 FPS Line (Real-time threshold)
    # We only show this if it fits reasonably, or we force the axis to include it
    fps_latency = 33.33
    ax.axvline(fps_latency, color='#FF5555', linestyle='--', linewidth=1.5, alpha=0.8, label='30 FPS', zorder=2)
    
    # 4. Styling & Limits
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy-Latency Trade-off')
    
    # FIX: Add margins so points/text don't hit the box edge
    ax.margins(x=0.15, y=0.15)
    
    # Optional: If data is far left (<10ms) and line is at 33ms, 
    # ensure 0 is included for perspective.
    ax.set_xlim(left=0, right=max(fps_latency + 5, max_latency * 1.2))

    # Legend
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, edgecolor='#CCCCCC')
    
    plt.tight_layout()
    
    # 5. Save
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for ext in ['pdf', 'png']:
        plt.savefig(output_path / f'fig1_pareto.{ext}', dpi=600 if ext=='png' else None)
    
    plt.close()
    print("   ✓ Saved")

def plot_topk(all_res, out_dir):
    print("\n📊 Top-k accuracy...")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    k_vals = list(range(1, 21))
    for name, res in all_res.items():
        ax.plot(k_vals, res['topk'], label=name.capitalize(),
               color=CONFIG['models'][name]['color'], marker=CONFIG['models'][name]['marker'],
               linewidth=2, markersize=4, markevery=2)
    
    ax.set_xlabel('k', fontweight='bold')
    ax.set_ylabel('Top-k (%)', fontweight='bold')
    ax.set_title('Top-k Accuracy', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim([1, 20])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'fig2_topk.{ext}', dpi=600 if ext=='png' else None)
    plt.close()
    print("   ✓ Saved")

def plot_f1_violin(all_res, out_dir):
    print("\n📊 Per-class F1 violin...")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    parts = ax.violinplot(
        [all_res[n]['per_class_f1'] for n in all_res.keys()],
        positions=range(len(all_res)),
        widths=0.6,
        showmeans=True,
        showmedians=True
    )
    
    for i, (pc, n) in enumerate(zip(parts['bodies'], all_res.keys())):
        pc.set_facecolor(CONFIG['models'][n]['color'])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    ax.set_xticks(range(len(all_res)))
    ax.set_xticklabels([n.capitalize() for n in all_res.keys()])
    ax.set_ylabel('Per-Class F1 (%)', fontweight='bold')
    ax.set_title('F1 Distribution', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'fig3_f1.{ext}', dpi=600 if ext=='png' else None)
    plt.close()
    print("   ✓ Saved")

def plot_confusion_diff(all_res, out_dir):
    print("\n📊 Confusion difference (base - tiny)...")
    
    if 'base' not in all_res or 'tiny' not in all_res:
        print("   ⚠️  Need both 'base' and 'tiny'")
        return
    
    b_pred, b_lab = all_res['base']['predictions'], all_res['base']['labels']
    t_pred, t_lab = all_res['tiny']['predictions'], all_res['tiny']['labels']
    
    # Top 50 classes
    unq, cnts = np.unique(b_lab, return_counts=True)
    top = unq[np.argsort(cnts)[-50:]]
    
    mask_b = np.isin(b_lab, top)
    mask_t = np.isin(t_lab, top)
    
    cm_b = confusion_matrix(b_lab[mask_b], b_pred[mask_b], labels=top)
    cm_t = confusion_matrix(t_lab[mask_t], t_pred[mask_t], labels=top)
    
    cm_b_norm = cm_b.astype('float') / (cm_b.sum(axis=1, keepdims=True) + 1e-10) * 100
    cm_t_norm = cm_t.astype('float') / (cm_t.sum(axis=1, keepdims=True) + 1e-10) * 100
    
    cm_diff = cm_b_norm - cm_t_norm
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    
    sns.heatmap(cm_b_norm, cmap='Blues', ax=axes[0], vmin=0, vmax=100,
               square=True, xticklabels=False, yticklabels=False,
               cbar_kws={'label': 'Acc (%)'})
    axes[0].set_title('Base', fontweight='bold')
    axes[0].set_ylabel('True', fontweight='bold')
    axes[0].set_xlabel('Pred', fontweight='bold')
    
    sns.heatmap(cm_diff, cmap='RdBu_r', ax=axes[1], center=0, vmin=-20, vmax=20,
               square=True, xticklabels=False, yticklabels=False,
               cbar_kws={'label': 'Diff (%)'})
    axes[1].set_title('Base − Tiny', fontweight='bold')
    axes[1].set_ylabel('True', fontweight='bold')
    axes[1].set_xlabel('Pred', fontweight='bold')
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'fig4_confusion_diff.{ext}', dpi=600 if ext=='png' else None)
    plt.close()
    print("   ✓ Saved")

def plot_calibration(all_res, out_dir):
    print("\n📊 Calibration...")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    bins = np.linspace(0, 1, 11)
    bin_ctrs = (bins[:-1] + bins[1:]) / 2
    
    for name, res in all_res.items():
        preds = res['predictions']
        labs = res['labels']
        probs = res['probabilities']
        
        corr = (preds == labs)
        confs = probs[np.arange(len(preds)), preds]
        
        bin_accs = []
        for i in range(len(bins) - 1):
            mask = (confs >= bins[i]) & (confs < bins[i+1])
            bin_accs.append(corr[mask].mean() if mask.sum() > 0 else np.nan)
        
        ax.plot(bin_ctrs, np.array(bin_accs) * 100,
               label=f"{name.capitalize()} (ECE: {res['ece']:.2f}%)",
               color=CONFIG['models'][name]['color'], marker=CONFIG['models'][name]['marker'],
               linewidth=2, markersize=5)
    
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Perfect')
    ax.set_xlabel('Confidence', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Calibration', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(out_dir / f'fig5_calibration.{ext}', dpi=600 if ext=='png' else None)
    plt.close()
    print("   ✓ Saved")

# ============================================================================
# TABLES
# ============================================================================

def generate_tables(all_res, out_dir):
    print("\n📄 Generating LaTeX tables...")
    
    # Summary table
    data = []
    for name, res in all_res.items():
        data.append({
            'Model': name.capitalize(),
            'Params': f"{res['params_M']:.2f}",
            'Size': f"{res['size_MB']:.2f}",
            'Latency': f"{res['latency_ms']:.2f}",
            'Throughput': f"{res['throughput']:.1f}",
            'Top-1': f"{res['top1']:.2f}",
            'Top-5': f"{res['top5']:.2f}",
        })
    
    df = pd.DataFrame(data)
    df.to_csv(out_dir / 'table1_summary.csv', index=False)
    
    # LaTeX
    best_top1 = df['Top-1'].astype(float).idxmax()
    best_top5 = df['Top-5'].astype(float).idxmax()
    
    latex = """\\begin{table}[t]
\\caption{Model comparison. Best in \\textbf{bold}.}
\\label{tab:summary}
\\centering\\footnotesize
\\begin{tabular}{lrrrrrrr}
\\toprule
Model & Params (M) & Size (MB) & Latency (ms) & Throughput (s/s) & Top-1 (\\%) & Top-5 (\\%) \\\\
\\midrule
"""
    
    for idx, row in df.iterrows():
        t1 = f"\\textbf{{{row['Top-1']}}}" if idx == best_top1 else row['Top-1']
        t5 = f"\\textbf{{{row['Top-5']}}}" if idx == best_top5 else row['Top-5']
        latex += f"{row['Model']} & {row['Params']} & {row['Size']} & {row['Latency']} & {row['Throughput']} & {t1} & {t5} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    with open(out_dir / 'table1_summary.tex', 'w') as f:
        f.write(latex)
    
    print("   ✓ Saved: table1_summary.csv & .tex")
    
    # Statistical tests
    if len(all_res) >= 2:
        print("\n📊 Statistical tests...")
        names = list(all_res.keys())
        stats_data = []
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                # Note: Requires multiple seeds - placeholder here
                diff = all_res[n2]['top1'] - all_res[n1]['top1']
                stats_data.append({
                    'Comparison': f"{n2.capitalize()} vs {n1.capitalize()}",
                    'Diff (%)': f"{diff:.2f}",
                    'p-value': 'N/A',  # Need multiple runs
                    'Effect': 'N/A'
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(out_dir / 'table2_stats.csv', index=False)
        print("   ✓ Saved: table2_stats.csv")

# ============================================================================
# MAIN
# ============================================================================

def main():
    config = CONFIG
    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}\nIEEE MODEL COMPARISON\n{'='*60}")
    print(f"Device: {device}\nOutput: {out_dir}\n")
    
    # Benchmark all models
    all_results = {}
    for name, info in config['models'].items():
        ckpt = info['checkpoint']
        if not Path(ckpt).exists():
            print(f"⚠️  Skipping {name}: checkpoint not found")
            continue
        
        all_results[name] = benchmark_model(name, ckpt, device, config)
    
    if len(all_results) < 2:
        print("\n❌ Need at least 2 models for comparison")
        return
    
    # Generate plots
    print(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}")
    plot_pareto(all_results, out_dir)
    plot_topk(all_results, out_dir)
    plot_f1_violin(all_results, out_dir)
    plot_confusion_diff(all_results, out_dir)
    plot_calibration(all_results, out_dir)
    
    # Generate tables
    generate_tables(all_results, out_dir)
    
    # Summary
    print(f"\n{'='*60}\n✅ COMPLETE\n{'='*60}")
    print(f"\nOutput: {out_dir}/")
    print("Figures: fig1_pareto, fig2_topk, fig3_f1, fig4_confusion_diff, fig5_calibration")
    print("Tables: table1_summary.csv/.tex, table2_stats.csv")
    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()