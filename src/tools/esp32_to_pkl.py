"""
esp32_to_pkl.py — Phase S-6 Converter
─────────────────────────────────────────────────────────
Convert ESP32 CSV data collection folder → .pkl session file
that main.py --mode load can visualize.

Usage:
    python tools/esp32_to_pkl.py --input csi_data --room 5x4x2.5
    # Then:
    python main.py --mode load --file esp32_session.pkl

Input folder structure (created by collect_data.py):
    csi_data/
        r00_c00.csv
        r00_c01.csv
        r01_c00.csv
        ...

Each CSV row format (from ESP32 firmware):
    CSI_DATA, timestamp_us, rssi_dBm, n_subcarriers, extra1, extra2, I0, Q0, I1, Q1, ...
─────────────────────────────────────────────────────────
"""

import os
import sys
import pickle
import argparse
import numpy as np

# ── Ensure project root is on path for pipeline imports ──────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.csi_parser import parse_line

def parse_csv_file(filepath: str) -> list:
    """Parse one grid-point CSV. Returns list of amplitude arrays."""
    amplitudes = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header row
            result = parse_line(line.strip())
            if result is None:
                continue
            amplitudes.append(result["amplitude"])
    except Exception as e:
        print(f"  [WARN] Could not parse {filepath}: {e}")
    return amplitudes


def compute_feature(amp_list: list) -> float:
    """Compute a single RF feature (variance of mean amplitude) for one grid point."""
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


def parse_room(room_str: str) -> tuple:
    """Parse '5x4x2.5' → (5.0, 4.0, 2.5)."""
    parts = room_str.lower().replace(" ", "").split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid room format '{room_str}'. Use WxDxH e.g. 5x4x2.5")
    return tuple(float(p) for p in parts)


def auto_detect_grid(csv_files: list) -> tuple:
    """Infer grid dimensions from filenames like r00_c00.csv."""
    max_r, max_c = 0, 0
    for fname in csv_files:
        base = os.path.basename(fname).replace(".csv", "")
        try:
            parts = base.split("_")
            r = int(parts[0][1:])
            c = int(parts[1][1:])
            max_r = max(max_r, r)
            max_c = max(max_c, c)
        except Exception:
            continue
    return max_r + 1, max_c + 1


def convert(input_folder: str, room_str: str, output_file: str):
    room_w, room_d, room_h = parse_room(room_str)

    # Find all CSV files
    csv_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".csv") and f.startswith("r") and "_c" in f
    ])

    if not csv_files:
        print(f"[CONVERT] No CSV files found in '{input_folder}'")
        print("[CONVERT] Expected files like r00_c00.csv, r00_c01.csv, ...")
        sys.exit(1)

    print(f"[CONVERT] Found {len(csv_files)} CSV files")

    rows, cols = auto_detect_grid(csv_files)
    print(f"[CONVERT] Detected grid: {rows} rows × {cols} cols")

    # Build grid + collect all frames per position
    grid_points = []
    total_frames = 0
    voxel_feature_grid = np.full((rows, cols), np.nan)

    for r in range(rows):
        for c in range(cols):
            fpath = os.path.join(input_folder, f"r{r:02d}_c{c:02d}.csv")
            if not os.path.exists(fpath):
                print(f"  [SKIP] Missing: {fpath}")
                continue

            amp_list = parse_csv_file(fpath)
            feature = compute_feature(amp_list)
            n_frames = len(amp_list)
            total_frames += n_frames

            # Real-world position (centre of grid cell)
            # TX is at (0, 0, 1.0) — corner, 1m height
            x_m = (c / max(cols - 1, 1)) * room_w
            y_m = (r / max(rows - 1, 1)) * room_d
            z_m = 1.0  # receiver height (fixed)

            voxel_feature_grid[r, c] = feature

            grid_points.append({
                "row": r,
                "col": c,
                "x_m": round(x_m, 3),
                "y_m": round(y_m, 3),
                "z_m": round(z_m, 3),
                "n_frames": n_frames,
                "feature": feature,
                "amp_list_mean": float(np.mean(amp_list)) if amp_list else 0.0,
            })

            print(f"  r={r} c={c} → ({x_m:.2f}, {y_m:.2f}, {z_m:.2f})m"
                  f"  --  {n_frames} frames  feature={feature:.4f}")

    # Build 3D voxel grid by extruding 2D
    nH = max(1, int(round(room_h / (room_d / max(rows, 1)))))
    g_norm = voxel_feature_grid.copy()
    finite = g_norm[np.isfinite(g_norm)]
    if len(finite) > 0:
        g_norm = np.where(np.isnan(g_norm), 0.0, g_norm)
        gmin, gmax = g_norm.min(), g_norm.max()
        g_norm = (g_norm - gmin) / (gmax - gmin + 1e-9)
    else:
        g_norm = np.zeros_like(g_norm)

    voxel_grid_3d = np.stack([g_norm] * nH, axis=-1)  # (rows, cols, nH)

    # Session dict — compatible with main.py --mode load
    session = {
        "source": "esp32",
        "room_w": room_w,
        "room_d": room_d,
        "room_h": room_h,
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_points": grid_points,
        "voxel_grid_3d": voxel_grid_3d,
        "voxel_feature_grid_2d": voxel_feature_grid,
        "total_frames": total_frames,
    }

    with open(output_file, "wb") as f:
        pickle.dump(session, f)

    print(f"\n[CONVERT] Total frames: {total_frames:,}")
    print(f"[CONVERT] Session saved to: {output_file}")
    print(f"\nNow visualize with:")
    print(f"  python main.py --mode load --file {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ESP32 CSI CSV collection to .pkl session for 3D visualization."
    )
    parser.add_argument("--input",  required=True, help="Folder containing r??_c??.csv files")
    parser.add_argument("--room",   required=True, help="Room dimensions WxDxH e.g. 5x4x2.5")
    parser.add_argument("--output", default="esp32_session.pkl", help="Output .pkl filename")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[CONVERT] Input folder not found: '{args.input}'")
        sys.exit(1)

    convert(args.input, args.room, args.output)


if __name__ == "__main__":
    main()
