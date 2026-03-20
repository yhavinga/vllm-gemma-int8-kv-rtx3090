#!/usr/bin/env python3
"""Generate grouped bar chart for 4B throughput.

Data source: docs/int8-kv-audit/data/throughput_4b_configs.csv
             research/14-throughput-grid-search.md
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "int8-kv-audit" / "plots"

# Data from CSV and research notes
contexts = ['4K', '8K', '16K', '32K', '64K', '128K']
configs = ['TP=1', 'TP=2', 'TP=2+INT8', 'DP=2', 'DP=2+INT8']

data = {
    'TP=1':      [4066, 4067, 4037, 4014, None, None],
    'TP=2':      [5612, 5600, 5572, 5371, 5216, 4741],
    'TP=2+INT8': [5559, 5513, 5533, 5404, 5108, 4711],
    'DP=2':      [7298, 7317, 7296, 7182, None, None],
    'DP=2+INT8': [None, None, None, 7435, 7545, 7254],
}

COLORS = {
    'TP=1': '#636EFA',       # Blue
    'TP=2': '#EF553B',       # Red
    'TP=2+INT8': '#FFA15A',  # Orange
    'DP=2': '#00CC96',       # Green
    'DP=2+INT8': '#AB63FA',  # Purple
}


def plot_grouped_bars():
    """Clean grouped bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(contexts))
    n_configs = len(configs)
    width = 0.15

    for i, config in enumerate(configs):
        values = data[config]
        # Replace None with 0 for plotting, we'll mark them separately
        plot_values = [v if v else 0 for v in values]
        offset = (i - n_configs/2 + 0.5) * width

        bars = ax.bar(x + offset, plot_values, width,
                     label=config, color=COLORS[config], alpha=0.9)

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val:
                height = bar.get_height()
                ax.annotate(f'{val//1000}.{(val%1000)//100}k',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=90)
            else:
                # Mark OOM/not-tested
                ax.annotate('OOM' if config in ['TP=1', 'DP=2'] else '',
                           xy=(bar.get_x() + bar.get_width()/2, 200),
                           ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Gemma 3 4B W4A16: Throughput by Configuration\n(DP=2+INT8 enables long context where others OOM)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.set_ylim(0, 8500)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation for key insight
    ax.annotate('INT8 enables\nDP at 64K+', xy=(4.3, 7545), fontsize=10,
               ha='center', color='#AB63FA', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_grouped_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_grouped_bars.png'}")


def plot_stacked_context_focus():
    """Alternative: Focus on the story - short vs long context."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Short context (4K-32K) - DP=2 wins
    short_contexts = ['4K', '8K', '16K', '32K']
    short_configs = ['TP=1', 'TP=2', 'DP=2']
    x = np.arange(len(short_contexts))
    width = 0.25

    for i, config in enumerate(short_configs):
        values = data[config][:4]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=config, color=COLORS[config])
        for bar, val in zip(bars, values):
            ax1.annotate(f'{val//1000}.{(val%1000)//100}k',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Context Length', fontsize=12)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax1.set_title('Short Context: DP=2 Wins (+30% over TP=2)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_contexts)
    ax1.set_ylim(0, 8500)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Right: Long context (64K-128K) - Only DP=2+INT8 and TP=2 work
    long_contexts = ['64K', '128K']
    long_configs = ['TP=2', 'DP=2+INT8']
    x = np.arange(len(long_contexts))
    width = 0.35

    for i, config in enumerate(long_configs):
        if config == 'TP=2':
            values = [data[config][4], data[config][5]]  # 64K, 128K
        else:
            values = [data[config][4], data[config][5]]
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, values, width, label=config, color=COLORS[config])
        for bar, val in zip(bars, values):
            ax2.annotate(f'{val:,}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Context Length', fontsize=12)
    ax2.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax2.set_title('Long Context: INT8 Enables DP (+45%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(long_contexts)
    ax2.set_ylim(0, 8500)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage annotations
    ax2.annotate('+45%', xy=(0, 6400), fontsize=14, fontweight='bold', color='#AB63FA', ha='center')
    ax2.annotate('+53%', xy=(1, 6000), fontsize=14, fontweight='bold', color='#AB63FA', ha='center')

    plt.suptitle('Gemma 3 4B: Why INT8 KV Cache Matters', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_short_vs_long.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_short_vs_long.png'}")


def plot_simple_comparison():
    """Simplest view: just the key comparison at each context length."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(contexts))
    width = 0.35

    # Best non-INT8 option at each context
    best_without_int8 = [7298, 7317, 7296, 7182, 5216, 4741]  # DP=2 short, TP=2 long
    labels_without = ['DP=2', 'DP=2', 'DP=2', 'DP=2', 'TP=2', 'TP=2']

    # Best with INT8
    best_with_int8 = [None, None, None, 7435, 7545, 7254]  # DP=2+INT8

    bars1 = ax.bar(x - width/2, best_without_int8, width,
                   label='Best without INT8', color='#636EFA', alpha=0.8)

    # Only plot INT8 where we have data
    int8_x = [3, 4, 5]
    int8_vals = [7435, 7545, 7254]
    bars2 = ax.bar([i + width/2 for i in int8_x], int8_vals, width,
                   label='DP=2 + INT8', color='#AB63FA', alpha=0.9)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Mark OOM region
    ax.axvspan(3.5, 5.5, alpha=0.1, color='red')
    ax.annotate('Without INT8:\nDP=2 OOMs here', xy=(4.5, 3000),
               fontsize=10, ha='center', color='red', alpha=0.7)

    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('INT8 KV Cache Unlocks Data Parallelism at Long Context',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.set_ylim(0, 8500)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_int8_unlock.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_int8_unlock.png'}")


if __name__ == '__main__':
    print("Generating bar chart visualizations...")
    plot_grouped_bars()
    plot_stacked_context_focus()
    plot_simple_comparison()
    print("Done!")
