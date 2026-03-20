#!/usr/bin/env python3
"""Generate 3D and heatmap visualizations for 4B throughput grid.

Data source: docs/int8-kv-audit/data/throughput_4b_configs.csv
             research/14-throughput-grid-search.md (TP=1 and DP=2 short context)
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "int8-kv-audit" / "plots"

# Data from throughput_4b_configs.csv and research/14-throughput-grid-search.md
contexts = ['4K', '8K', '16K', '32K', '64K', '128K']
configs = ['TP=1', 'TP=2', 'TP=2+INT8', 'DP=2', 'DP=2+INT8']

# Throughput matrix [config][context] - None means OOM or not tested
data = {
    'TP=1':      [4066, 4067, 4037, 4014, None, None],      # OOM at 64K+
    'TP=2':      [5612, 5600, 5572, 5371, 5216, 4741],
    'TP=2+INT8': [5559, 5513, 5533, 5404, 5108, 4711],
    'DP=2':      [7298, 7317, 7296, 7182, None, None],      # OOM at 64K+
    'DP=2+INT8': [None, None, None, 7435, 7545, 7254],      # Only tested 32K+
}

COLORS = {
    'TP=1': '#636EFA',
    'TP=2': '#EF553B',
    'TP=2+INT8': '#FFA15A',
    'DP=2': '#00CC96',
    'DP=2+INT8': '#AB63FA',
}


def plot_3d_bars():
    """Create a 3D bar chart."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_pos = np.arange(len(contexts))
    y_pos = np.arange(len(configs))

    for yi, config in enumerate(configs):
        xs = []
        zs = []
        for xi, val in enumerate(data[config]):
            if val is not None:
                xs.append(xi)
                zs.append(val)

        if xs:
            ax.bar3d(xs, [yi] * len(xs), [0] * len(xs),
                    dx=0.6, dy=0.6, dz=zs,
                    color=COLORS[config], alpha=0.85, label=config)

    ax.set_xlabel('Context Length', fontsize=12, labelpad=10)
    ax.set_ylabel('Configuration', fontsize=12, labelpad=10)
    ax.set_zlabel('Throughput (tok/s)', fontsize=12, labelpad=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(contexts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontsize=9)

    ax.set_zlim(0, 8000)
    ax.view_init(elev=25, azim=-60)

    ax.set_title('Gemma 3 4B Throughput: DP+INT8 Dominates Long Context',
                fontsize=14, fontweight='bold', pad=20)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[c], label=c) for c in configs]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_3d.png'}")


def plot_heatmap():
    """Create a heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create matrix with NaN for missing values
    matrix = np.array([[v if v else np.nan for v in data[c]] for c in configs])

    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=4000, vmax=8000)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Throughput (tok/s)', fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(contexts)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(contexts, fontsize=11)
    ax.set_yticklabels(configs, fontsize=11)

    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Configuration', fontsize=12)
    ax.set_title('Gemma 3 4B Throughput Heatmap\n(Green = faster, Red = slower, Gray = OOM/not tested)',
                fontsize=13, fontweight='bold')

    # Add text annotations
    for i, config in enumerate(configs):
        for j, val in enumerate(data[config]):
            if val is not None:
                # White text on dark, black on light
                color = 'white' if val < 5500 else 'black'
                ax.text(j, i, f'{val:,}', ha='center', va='center',
                       color=color, fontsize=9, fontweight='bold')
            else:
                ax.text(j, i, 'OOM' if config in ['TP=1', 'DP=2'] else 'n/t',
                       ha='center', va='center', color='gray', fontsize=9)

    # Highlight best values per column
    for j in range(len(contexts)):
        col_vals = [(i, data[c][j]) for i, c in enumerate(configs) if data[c][j] is not None]
        if col_vals:
            best_i, best_val = max(col_vals, key=lambda x: x[1])
            # Add star
            ax.text(j, best_i, f'{best_val:,}', ha='center', va='center',
                   color='black', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_heatmap.png'}")


def plot_surface():
    """Create a 3D surface plot (interpolated)."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid
    x = np.arange(len(contexts))
    y = np.arange(len(configs))
    X, Y = np.meshgrid(x, y)

    # Create Z with interpolation for missing values
    Z = np.array([[v if v else np.nan for v in data[c]] for c in configs], dtype=float)

    # Plot surface with custom colormap
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                          linewidth=0.5, edgecolor='white', antialiased=True)

    # Mark actual data points
    for yi, config in enumerate(configs):
        for xi, val in enumerate(data[config]):
            if val is not None:
                ax.scatter([xi], [yi], [val], color='red', s=50, zorder=5)

    ax.set_xlabel('Context Length', fontsize=12, labelpad=10)
    ax.set_ylabel('Configuration', fontsize=12, labelpad=10)
    ax.set_zlabel('Throughput (tok/s)', fontsize=12, labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.set_yticks(y)
    ax.set_yticklabels(configs, fontsize=9)

    ax.set_zlim(3500, 8000)
    ax.view_init(elev=30, azim=-45)

    ax.set_title('Gemma 3 4B Throughput Surface\n(Red dots = measured, surface = interpolated)',
                fontsize=14, fontweight='bold', pad=20)

    fig.colorbar(surf, ax=ax, shrink=0.5, label='Throughput (tok/s)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_surface.png'}")


def plot_grouped_3d():
    """Create a cleaner 3D grouped bar chart."""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Positions
    x_pos = np.arange(len(contexts))
    width = 0.15
    depth = 0.5

    for idx, config in enumerate(configs):
        xs = []
        ys = []
        zs = []
        for xi, val in enumerate(data[config]):
            if val is not None:
                xs.append(xi + idx * width - 0.3)
                ys.append(0)
                zs.append(val)

        if xs:
            ax.bar3d(xs, ys, [0]*len(xs),
                    dx=width*0.9, dy=depth, dz=zs,
                    color=COLORS[config], alpha=0.9,
                    shade=True)

    ax.set_xlabel('\nContext Length', fontsize=12)
    ax.set_zlabel('Throughput (tok/s)', fontsize=12)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(contexts)
    ax.set_yticks([])

    ax.set_zlim(0, 8500)
    ax.view_init(elev=20, azim=-70)

    ax.set_title('Gemma 3 4B: Throughput by Config and Context Length',
                fontsize=14, fontweight='bold', pad=20)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[c], label=c) for c in configs]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
             bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_4b_3d_grouped.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'throughput_4b_3d_grouped.png'}")


if __name__ == '__main__':
    print("Generating 3D and heatmap plots...")
    plot_3d_bars()
    plot_heatmap()
    plot_surface()
    plot_grouped_3d()
    print("Done! Check which visualization you prefer.")
