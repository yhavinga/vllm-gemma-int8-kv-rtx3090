#!/usr/bin/env python3
"""Generate the definitive per-layer scales visualization.

Data source: docs/int8-kv-audit/data/per_layer_scales_gemma3_27b_tp2.csv

Key insight: V values vary 340x across 62 layers. A global scale wastes
63% of quantization budget on low-magnitude layers.
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "docs" / "int8-kv-audit" / "data" / "per_layer_scales_gemma3_27b_tp2.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "int8-kv-audit" / "plots"

# Gemma 3 27B global attention layers (from architecture)
GLOBAL_ATTENTION_LAYERS = {24, 25, 49}  # Every 6th layer after layer 24, but also special ones


def load_data():
    """Load per-layer scale data from CSV."""
    layers = []
    k_absmax = []
    v_absmax = []
    k_scale = []
    v_scale = []

    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layers.append(int(row['layer_idx']))
            k_absmax.append(float(row['k_absmax_seen']))
            v_absmax.append(float(row['v_absmax_seen']))
            k_scale.append(float(row['k_scale']))
            v_scale.append(float(row['v_scale']))

    return {
        'layers': np.array(layers),
        'k_absmax': np.array(k_absmax),
        'v_absmax': np.array(v_absmax),
        'k_scale': np.array(k_scale),
        'v_scale': np.array(v_scale),
    }


def plot_comprehensive():
    """The main visualization: dual-axis plot showing the 340x problem."""
    data = load_data()

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot K absmax on left axis (blue)
    color_k = '#636EFA'
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('K absmax', color=color_k, fontsize=12)
    line_k = ax1.plot(data['layers'], data['k_absmax'], 'o-', color=color_k,
                      label='K absmax (5.8x range)', markersize=4, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_k)
    ax1.set_ylim(0, 80)

    # Plot V absmax on right axis (orange) - LOG SCALE to show variance
    ax2 = ax1.twinx()
    color_v = '#EF553B'
    ax2.set_ylabel('V absmax (log scale)', color=color_v, fontsize=12)
    line_v = ax2.plot(data['layers'], data['v_absmax'], 's-', color=color_v,
                      label='V absmax (340x range)', markersize=4, linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_v)
    ax2.set_yscale('log')
    ax2.set_ylim(1, 1000)

    # Highlight extreme layers
    max_v_idx = np.argmax(data['v_absmax'])
    min_v_idx = np.argmin(data['v_absmax'])

    ax2.annotate(f"Layer {data['layers'][max_v_idx]}\nv_absmax={data['v_absmax'][max_v_idx]:.0f}",
                xy=(data['layers'][max_v_idx], data['v_absmax'][max_v_idx]),
                xytext=(data['layers'][max_v_idx] - 8, 600),
                fontsize=10, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    ax2.annotate(f"Layer {data['layers'][min_v_idx]}\nv_absmax={data['v_absmax'][min_v_idx]:.1f}",
                xy=(data['layers'][min_v_idx], data['v_absmax'][min_v_idx]),
                xytext=(data['layers'][min_v_idx] - 8, 8),
                fontsize=10, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    # Add horizontal line showing global scale problem
    global_v_max = data['v_absmax'].max()
    ax2.axhline(y=global_v_max, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('Global scale optimized for max',
                xy=(50, global_v_max), fontsize=9, color='gray', ha='center')

    # Title and legend
    plt.title('Gemma 3 27B: Per-Layer K/V Activation Ranges\n'
              'V varies 340x across layers — global scale wastes quantization budget',
              fontsize=14, fontweight='bold')

    # Combined legend
    lines = line_k + line_v
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 62)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_layer_scales_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_layer_scales_comprehensive.png'}")


def plot_wasted_budget():
    """Show how much quantization budget is wasted with global vs per-layer."""
    data = load_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: V absmax with global scale line
    global_v_scale = data['v_absmax'].max() / 127  # Scale for global
    effective_levels = (data['v_absmax'] / data['v_absmax'].max()) * 127

    colors = ['#00CC96' if lvl > 64 else '#FFA15A' if lvl > 32 else '#EF553B'
              for lvl in effective_levels]

    ax1.bar(data['layers'], effective_levels, color=colors, width=0.8)
    ax1.axhline(y=127, color='black', linestyle='-', linewidth=2)
    ax1.axhline(y=64, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('Full INT8 range (127)', xy=(55, 130), fontsize=10)
    ax1.annotate('50% utilization', xy=(55, 67), fontsize=9, color='gray')

    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Effective INT8 levels used', fontsize=12)
    ax1.set_title('Global Scale: Many Layers Waste Quantization Budget',
                 fontsize=12, fontweight='bold')
    ax1.set_xlim(-1, 62)
    ax1.set_ylim(0, 145)

    # Highlight worst case
    min_idx = np.argmin(effective_levels)
    ax1.annotate(f'Layer {min_idx}\nOnly {effective_levels[min_idx]:.0f}/127 levels\n= {100*effective_levels[min_idx]/127:.0f}% utilized',
                xy=(min_idx, effective_levels[min_idx]),
                xytext=(min_idx + 5, 50),
                fontsize=10, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    # Right: Per-layer scale gives full utilization
    ax2.bar(data['layers'], [127] * len(data['layers']), color='#00CC96', width=0.8, alpha=0.9)
    ax2.axhline(y=127, color='black', linestyle='-', linewidth=2)

    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Effective INT8 levels used', fontsize=12)
    ax2.set_title('Per-Layer Scale: Every Layer Uses Full Range',
                 fontsize=12, fontweight='bold')
    ax2.set_xlim(-1, 62)
    ax2.set_ylim(0, 145)
    ax2.annotate('All 127 levels utilized', xy=(31, 135), fontsize=11,
                ha='center', fontweight='bold', color='#00CC96')

    plt.suptitle('Why Per-Layer Quantization Scales Matter', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_layer_scales_wasted_budget.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_layer_scales_wasted_budget.png'}")


def plot_k_vs_v_scatter():
    """Scatter plot showing K is stable, V is wild."""
    data = load_data()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by layer index
    scatter = ax.scatter(data['k_absmax'], data['v_absmax'],
                        c=data['layers'], cmap='viridis', s=80, alpha=0.8)

    # Mark extremes
    max_v_idx = np.argmax(data['v_absmax'])
    min_v_idx = np.argmin(data['v_absmax'])
    max_k_idx = np.argmax(data['k_absmax'])

    ax.scatter([data['k_absmax'][max_v_idx]], [data['v_absmax'][max_v_idx]],
              color='red', s=200, marker='*', zorder=5, label=f'Max V (layer {max_v_idx})')
    ax.scatter([data['k_absmax'][min_v_idx]], [data['v_absmax'][min_v_idx]],
              color='blue', s=200, marker='*', zorder=5, label=f'Min V (layer {min_v_idx})')

    ax.set_xlabel('K absmax', fontsize=12)
    ax.set_ylabel('V absmax (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('K vs V Activation Ranges: V Has 340x Variance\n'
                 'Each point is one attention layer (62 total)',
                fontsize=13, fontweight='bold')

    # Add ratio annotation
    ax.annotate(f'V range: {data["v_absmax"].min():.1f} - {data["v_absmax"].max():.0f} (340x)\n'
               f'K range: {data["k_absmax"].min():.1f} - {data["k_absmax"].max():.1f} (5.8x)',
               xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.colorbar(scatter, label='Layer Index')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_layer_scales_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_layer_scales_scatter.png'}")


def plot_histogram():
    """Show distribution of K and V scales."""
    data = load_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # K absmax histogram
    ax1.hist(data['k_absmax'], bins=20, color='#636EFA', edgecolor='white', alpha=0.8)
    ax1.axvline(data['k_absmax'].mean(), color='red', linestyle='--',
               label=f'Mean: {data["k_absmax"].mean():.1f}')
    ax1.set_xlabel('K absmax', fontsize=12)
    ax1.set_ylabel('Count (layers)', fontsize=12)
    ax1.set_title(f'K Distribution: Tight (5.8x range)', fontsize=12, fontweight='bold')
    ax1.legend()

    # V absmax histogram (log-spaced bins)
    bins = np.logspace(np.log10(data['v_absmax'].min()), np.log10(data['v_absmax'].max()), 20)
    ax2.hist(data['v_absmax'], bins=bins, color='#EF553B', edgecolor='white', alpha=0.8)
    ax2.set_xscale('log')
    ax2.axvline(data['v_absmax'].mean(), color='red', linestyle='--',
               label=f'Mean: {data["v_absmax"].mean():.0f}')
    ax2.axvline(data['v_absmax'].max(), color='darkred', linestyle='-',
               label=f'Max: {data["v_absmax"].max():.0f}')
    ax2.set_xlabel('V absmax (log scale)', fontsize=12)
    ax2.set_ylabel('Count (layers)', fontsize=12)
    ax2.set_title(f'V Distribution: Spread (340x range)', fontsize=12, fontweight='bold')
    ax2.legend()

    plt.suptitle('Gemma 3 27B: K is Stable, V is Wildly Variable', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_layer_scales_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_layer_scales_histogram.png'}")


def plot_hero_comparison():
    """The hero figure: side-by-side showing the problem and solution."""
    data = load_data()

    fig = plt.figure(figsize=(16, 7))

    # Create custom grid: main plot on left, annotation on right
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

    ax_main = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    # Main plot: bar chart with both K and V
    x = data['layers']
    width = 0.4

    bars_k = ax_main.bar(x - width/2, data['k_absmax'], width, label='K absmax → INT8',
                         color='#636EFA', alpha=0.8)
    bars_v = ax_main.bar(x + width/2, data['v_absmax'], width, label='V absmax → FP8-E4M3',
                         color='#EF553B', alpha=0.8)

    ax_main.set_xlabel('Layer Index', fontsize=12)
    ax_main.set_ylabel('Absolute Maximum Activation', fontsize=12)
    ax_main.set_title('Per-Layer K/V Activation Ranges (Gemma 3 27B, 62 layers)',
                     fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=11)
    ax_main.set_xlim(-1, 62)
    ax_main.set_yscale('symlog', linthresh=100)  # Symlog to show both small and large

    # Mark layer 42 and 59
    ax_main.annotate(f'Layer 42\nV={data["v_absmax"][42]:.0f}',
                    xy=(42, data['v_absmax'][42]),
                    xytext=(42, 1200),
                    fontsize=10, fontweight='bold', color='darkred', ha='center',
                    arrowprops=dict(arrowstyle='->', color='darkred'))

    ax_main.annotate(f'Layer 59\nV={data["v_absmax"][59]:.1f}',
                    xy=(59, data['v_absmax'][59]),
                    xytext=(55, 15),
                    fontsize=10, fontweight='bold', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred'))

    # Stats panel
    ax_stats.axis('off')

    stats_text = """
    THE 340x PROBLEM

    V absmax range:
    • Min: 2.6 (layer 59)
    • Max: 884 (layer 42)
    • Ratio: 340x

    K absmax range:
    • Min: 12.4 (layer 38)
    • Max: 71.5 (layer 29)
    • Ratio: 5.8x

    ─────────────────────

    SOLUTION: HYBRID

    K cache: INT8
    • Stable range (5.8x)
    • Linear quantization
    • Uniform spacing OK

    V cache: FP8-E4M3
    • Wild range (340x)
    • Log quantization
    • Handles outliers

    Both use per-layer
    scales (496 bytes)
    """

    ax_stats.text(0.05, 0.98, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_layer_scales_hero.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_layer_scales_hero.png'}")


if __name__ == '__main__':
    print("Generating per-layer scale visualizations...")
    print(f"Reading data from: {DATA_FILE}")

    plot_comprehensive()      # Dual-axis line plot
    plot_wasted_budget()      # Side-by-side global vs per-layer
    plot_k_vs_v_scatter()     # Scatter showing variance
    plot_histogram()          # Distribution comparison
    plot_hero_comparison()    # The definitive hero figure

    print("Done!")
