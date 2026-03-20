#!/usr/bin/env python3
"""Generate plots for the INT8 KV cache paper.

Data sources:
- 4B results: docs/int8-kv-audit/data/throughput_4b_configs.csv
- 1B results: research/14-throughput-grid-search.md (lines 63-77)
- 27B results: research/11-int8-kv-cache.md (lines 162-163), twitter-thread.md
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

# Project root and output directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "int8-kv-audit" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_4b_throughput_csv():
    """Load 4B throughput data from CSV."""
    csv_path = PROJECT_ROOT / "docs" / "int8-kv-audit" / "data" / "throughput_4b_configs.csv"
    data = {'context': [], 'tp2_bf16': [], 'tp2_int8': [], 'dp2_bf16': [], 'dp2_int8': []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['context'].append(int(row['context_len']) // 1024)  # Convert to K
            data['tp2_bf16'].append(float(row['tp2_bf16']) if row['tp2_bf16'] else None)
            data['tp2_int8'].append(float(row['tp2_int8']) if row['tp2_int8'] else None)
            data['dp2_bf16'].append(float(row['dp2_bf16']) if row['dp2_bf16'] else None)
            data['dp2_int8'].append(float(row['dp2_int8']) if row['dp2_int8'] else None)

    return data

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'tp1': '#636EFA',      # Blue
    'tp2': '#EF553B',      # Red
    'tp2_int8': '#FFA15A', # Orange
    'dp2': '#00CC96',      # Green
    'dp2_int8': '#AB63FA', # Purple
    'bf16': '#19D3F3',     # Cyan
    'int8': '#FF6692',     # Pink
}


def plot_1b_throughput():
    """Plot 1B W8A8 throughput comparison.

    Data source: research/14-throughput-grid-search.md
    - TP=1: lines 63-68
    - TP=2: lines 72-77
    - DP=2: lines 37-42
    """
    contexts = ['4K', '8K', '16K', '32K']
    x = np.arange(len(contexts))
    width = 0.25

    # From research/14-throughput-grid-search.md
    tp1 = [8069, 8109, 8075, 7949]   # lines 65-68
    tp2 = [7148, 7179, 7091, 6848]   # lines 74-77
    dp2 = [12234, 12513, 12161, 11975]  # lines 39-42

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, tp1, width, label='TP=1 (single GPU)', color=COLORS['tp1'])
    bars2 = ax.bar(x, tp2, width, label='TP=2 (slower!)', color=COLORS['tp2'])
    bars3 = ax.bar(x + width, dp2, width, label='DP=2 (+51%)', color=COLORS['dp2'])

    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Gemma 3 1B W8A8: Data Parallelism Beats Tensor Parallelism', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 14000)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    # Add annotation about TP=2 being slower
    ax.annotate('NVLink overhead\nexceeds benefit',
               xy=(1, 7179), xytext=(1.5, 9500),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_1b_configs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_1b_configs.png'}")


def plot_27b_int8_comparison():
    """Plot 27B BF16 vs INT8 KV cache comparison.

    Data source: research/11-int8-kv-cache.md (lines 162-163)
                 twitter-thread.md (lines 37-42)
                 README.md (lines 9-10)
    """
    categories = ['Short\n(<4K)', '7K\ntokens', '12K\ntokens']
    x = np.arange(len(categories))
    width = 0.35

    # From research/11-int8-kv-cache.md and twitter-thread.md
    bf16 = [67, 24, 24]   # BF16 baseline
    int8 = [61, 45, 40]   # INT8 per-layer
    changes = ['-9%', '+87%', '+67%']

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, bf16, width, label='BF16 KV Cache', color=COLORS['bf16'])
    bars2 = ax.bar(x + width/2, int8, width, label='INT8 KV Cache (per-layer)', color=COLORS['int8'])

    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Gemma 3 27B: INT8 KV Cache Wins at Long Context', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 80)

    # Add value labels with change percentages
    for i, (bar1, bar2, change) in enumerate(zip(bars1, bars2, changes)):
        ax.annotate(f'{int(bar1.get_height())}',
                   xy=(bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

        color = '#00CC96' if change.startswith('+') else '#EF553B'
        ax.annotate(f'{int(bar2.get_height())}\n({change})',
                   xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    # Add crossover annotation
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('~4K crossover point',
               xy=(0.5, 55), fontsize=10, color='gray', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_27b_bf16_vs_int8.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_27b_bf16_vs_int8.png'}")


def plot_context_memory_comparison():
    """Plot max context: BF16 vs INT8."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['BF16 KV Cache', 'INT8 KV Cache']
    max_context = [32, 128]
    colors = [COLORS['bf16'], COLORS['int8']]

    bars = ax.bar(categories, max_context, color=colors, width=0.5)

    ax.set_ylabel('Maximum Context Length (K tokens)', fontsize=12)
    ax.set_title('INT8 KV Cache: 4x Maximum Context on Same Hardware', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 150)

    # Add value labels
    for bar, val in zip(bars, max_context):
        ax.annotate(f'{val}K',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Add 4x annotation
    ax.annotate('4x', xy=(0.5, 80), fontsize=24, fontweight='bold',
               ha='center', color='#00CC96')
    ax.annotate('', xy=(1, 128), xytext=(0, 32),
               arrowprops=dict(arrowstyle='->', color='#00CC96', lw=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'max_context_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'max_context_comparison.png'}")


def plot_kv_memory_savings():
    """Plot KV cache memory comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['BF16 KV Cache', 'INT8 KV Cache']
    memory = [23, 11.5]
    colors = [COLORS['bf16'], COLORS['int8']]

    bars = ax.bar(categories, memory, color=colors, width=0.5)

    ax.set_ylabel('KV Cache Memory (GB)', fontsize=12)
    ax.set_title('INT8 KV Cache: 50% Memory Reduction', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 28)

    # Add value labels
    for bar, val in zip(bars, memory):
        ax.annotate(f'{val} GB',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add savings annotation
    ax.annotate('-50%', xy=(0.5, 17), fontsize=20, fontweight='bold',
               ha='center', color='#00CC96')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kv_memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'kv_memory_comparison.png'}")


def plot_summary_hero():
    """Create a hero summary plot with all key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Plot 1: Speed improvement at long context
    ax1 = axes[0]
    metrics = ['Before\n(BF16)', 'After\n(INT8)']
    values = [24, 45]
    colors = ['#636EFA', '#00CC96']
    bars = ax1.bar(metrics, values, color=colors, width=0.6)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=11)
    ax1.set_title('Long Context Speed\n(7K tokens)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 55)
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')
    ax1.annotate('+87%', xy=(0.5, 35), fontsize=18, fontweight='bold', ha='center', color='#00CC96')

    # Plot 2: Max context
    ax2 = axes[1]
    metrics = ['Before\n(BF16)', 'After\n(INT8)']
    values = [32, 128]
    bars = ax2.bar(metrics, values, color=colors, width=0.6)
    ax2.set_ylabel('Max Context (K tokens)', fontsize=11)
    ax2.set_title('Maximum Context\nLength', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 150)
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val}K', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')
    ax2.annotate('4x', xy=(0.5, 80), fontsize=18, fontweight='bold', ha='center', color='#00CC96')

    # Plot 3: Memory savings
    ax3 = axes[2]
    metrics = ['Before\n(BF16)', 'After\n(INT8)']
    values = [23, 11.5]
    bars = ax3.bar(metrics, values, color=colors, width=0.6)
    ax3.set_ylabel('KV Cache Memory (GB)', fontsize=11)
    ax3.set_title('Memory\nUsage', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 28)
    for bar, val in zip(bars, values):
        ax3.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')
    ax3.annotate('-50%', xy=(0.5, 17), fontsize=18, fontweight='bold', ha='center', color='#00CC96')

    plt.suptitle('Gemma 3 27B on 2x RTX 3090: INT8 KV Cache Impact', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_hero.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'summary_hero.png'}")


def plot_speedup_journey():
    """Plot the optimization journey from 11 to 67 tok/s."""
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = ['Baseline\n(vLLM default)', 'CUDA graphs\n(--disable-custom-all-reduce)', 'INT8 KV\n(long context)']
    short_context = [11, 67, 61]  # -9% at short
    long_context = [11, 24, 45]   # +87% at long

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(x - width/2, short_context, width, label='Short context (<4K)', color=COLORS['tp1'])
    bars2 = ax.bar(x + width/2, long_context, width, label='Long context (7K)', color=COLORS['dp2'])

    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Optimization Journey: 11 → 67 tok/s (6x speedup)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 80)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add speedup annotations
    ax.annotate('6x', xy=(0.5, 50), fontsize=16, fontweight='bold', color='#636EFA')
    ax.annotate('+87%', xy=(1.7, 35), fontsize=14, fontweight='bold', color='#00CC96')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimization_journey.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'optimization_journey.png'}")


if __name__ == '__main__':
    print("Generating paper plots...")
    plot_1b_throughput()
    plot_27b_int8_comparison()
    plot_context_memory_comparison()
    plot_kv_memory_savings()
    plot_summary_hero()
    plot_speedup_journey()
    print("Done!")
