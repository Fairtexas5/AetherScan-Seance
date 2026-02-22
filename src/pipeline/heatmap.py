"""
heatmap.py
Processes CSI CSV files and generates heatmaps.
Supports both full processing and live partial preview after each grid point.
No API key required.

Also launches the interactive Plotly 3D viewer after the final heatmap is saved.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for automation
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel

from pipeline.csi_parser import parse_line

def parse_csv_file(filepath: str) -> list:
    """Load one grid point CSV and return list of amplitude arrays."""
    amplitudes = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            parsed = parse_line(line.strip())
            if parsed is None:
                continue
            amplitudes.append(parsed["amplitude"])
    except Exception as e:
        print(f"[heatmap] Error reading {filepath}: {e}")
    return amplitudes


def compute_feature(amp_list: list) -> float:
    """Compute single RF feature value from list of amplitude arrays."""
    if not amp_list:
        return 0.0
    mat = np.array(amp_list)
    mean_per_sub = np.mean(mat, axis=0)
    lo = np.percentile(mean_per_sub, 10)
    hi = np.percentile(mean_per_sub, 90)
    trimmed = mean_per_sub[(mean_per_sub >= lo) & (mean_per_sub <= hi)]
    if len(trimmed) == 0:
        return float(np.var(mean_per_sub))
    return float(np.var(trimmed))


def build_grid(data_folder: str, grid_rows: int, grid_cols: int) -> np.ndarray:
    """Load all CSV files and build the feature grid."""
    grid = np.full((grid_rows, grid_cols), np.nan)
    for r in range(grid_rows):
        for c in range(grid_cols):
            fpath = os.path.join(data_folder, f"r{r:02d}_c{c:02d}.csv")
            if os.path.exists(fpath):
                amp_list = parse_csv_file(fpath)
                grid[r, c] = compute_feature(amp_list)
    return grid


def render_heatmap(
    grid: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    room_width_m: float,
    room_height_m: float,
    output_path: str,
    title_suffix: str = ""
):
    """
    Render and save a heatmap PNG from a feature grid.
    NaN cells (not yet collected) shown in grey.
    """
    # Fill NaN with 0 for display (shown as dark)
    grid_display = np.where(np.isnan(grid), 0.0, grid)

    # Normalize
    gmin = np.nanmin(grid_display)
    gmax = np.nanmax(grid_display)
    grid_norm = (grid_display - gmin) / (gmax - gmin + 1e-9)

    # Smooth only collected areas
    collected_mask = ~np.isnan(grid)
    grid_smooth = gaussian_filter(grid_norm, sigma=1.0)

    # Edge detection
    ex = sobel(grid_smooth, axis=1)
    ey = sobel(grid_smooth, axis=0)
    edges = np.hypot(ex, ey)
    if edges.max() > 0:
        edges = edges / edges.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    collected_count = int(np.sum(collected_mask))
    total = grid_rows * grid_cols
    fig.suptitle(
        f"WiFi CSI Room RF Map {title_suffix}  [{collected_count}/{total} points collected]",
        fontsize=14, fontweight="bold"
    )

    # Left: heatmap
    im = axes[0].imshow(
        grid_smooth, cmap="hot", aspect="auto",
        extent=[0, room_width_m*100, room_height_m*100, 0],
        interpolation="bilinear", vmin=0, vmax=1
    )
    axes[0].set_title("RF Attenuation Map (CSI Variance)")
    axes[0].set_xlabel("Room Width (cm)")
    axes[0].set_ylabel("Room Depth (cm)")
    plt.colorbar(im, ax=axes[0], label="Normalized CSI Variance")

    # Grey overlay for uncollected points
    if not collected_mask.all():
        uncollected = np.where(collected_mask, np.nan, 0.5)
        axes[0].imshow(
            uncollected, cmap="gray", aspect="auto", alpha=0.5,
            extent=[0, room_width_m*100, room_height_m*100, 0]
        )

    # Right: edge detection
    axes[1].imshow(
        edges, cmap="Blues", aspect="auto",
        extent=[0, room_width_m*100, room_height_m*100, 0],
        interpolation="bilinear"
    )
    axes[1].set_title("Edge Detection (Estimated Boundaries)")
    axes[1].set_xlabel("Room Width (cm)")
    axes[1].set_ylabel("Room Depth (cm)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] Saved → {output_path}")


def live_preview(
    data_folder: str,
    grid_rows: int,
    grid_cols: int,
    room_width_m: float,
    room_height_m: float,
    output_folder: str,
    label: str = "preview"
):
    """
    Called after each grid point collection to regenerate the partial heatmap.
    Saves as output/preview_latest.png (overwritten each time).
    """
    grid = build_grid(data_folder, grid_rows, grid_cols)
    output_path = os.path.join(output_folder, "preview_latest.png")
    render_heatmap(grid, grid_rows, grid_cols, room_width_m, room_height_m, output_path, title_suffix=f"[{label}]")


def final_heatmap(
    data_folder: str,
    grid_rows: int,
    grid_cols: int,
    room_width_m: float,
    room_height_m: float,
    output_folder: str
):
    """Generate and save the final complete heatmap (2D PNG + interactive 3D Plotly viewer)."""
    grid = build_grid(data_folder, grid_rows, grid_cols)
    output_path = os.path.join(output_folder, "room_heatmap_final.png")
    render_heatmap(grid, grid_rows, grid_cols, room_width_m, room_height_m, output_path, title_suffix="[FINAL]")

    # ── Launch interactive 3D Plotly viewer ────────────────────────
    try:
        from pipeline.visualizer_3d import launch_from_grid2d
        print("[heatmap] Launching 3D interactive viewer...")
        # room_width_m = floor X dimension, room_height_m = floor Y (depth) dimension
        # ceiling height is estimated as 2.5m (standard room height)
        CEILING_H = 2.5
        launch_from_grid2d(
            grid2d=grid,
            room_w=room_width_m,
            room_d=room_height_m,   # room_height_m in CONFIG = floor depth (Y axis)
            room_h=CEILING_H,
            output_folder=output_folder,
            title="WiFi CSI 3D Room Map — Live Collection",
        )
    except Exception as e:
        print(f"[heatmap] 3D viewer skipped: {e}")
        print("[heatmap] Install plotly: pip install plotly")

    return output_path
