"""
generate_chair_csv.py
─────────────────────────────────────────────────────────
Generate synthetic CSI CSV files for the chair demo.

Grid: 6 rows × 4 cols
  Cell width : 27 cm  (measured from photos)
  Cell depth : 22 cm  (measured from photos)

Files are saved as:
  csi_data_chair/r00_c00.csv
  csi_data_chair/r00_c01.csv
  ...
  csi_data_chair/r05_c03.csv

Each CSV has 200 packets in the real firmware format:
  CSI_DATA,{timestamp_us},{rssi},{n_sub},0,0,{I0},{Q0},...

Run:
  python src/generate_chair_csv.py
─────────────────────────────────────────────────────────
"""

import os
import random
import math
import numpy as np

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "csi_data_chair")
PACKETS        = 200      # packets per grid position
N_SUB          = 64       # subcarriers (real ESP32 default)
BASE_AMP       = 22.0     # baseline IQ amplitude (quiet open space)
BASE_RSSI      = -52      # baseline RSSI (dBm)
CELL_W_CM      = 27       # measured grid cell width
CELL_D_CM      = 22       # measured grid cell depth
START_TS       = 23_500_000_000  # starting timestamp (µs)
PKT_INTERVAL   = 100_000  # ~10 packets/second

# ── CSI variance table (matches chair_demo.py) ─────────────────────
CSI_VARIANCE = np.array([
    # Col:  0      1      2      3
    [0.45,  0.50,  0.48,  0.38],   # Row 0 — clear LOS to TX
    [0.58,  0.88,  0.82,  0.44],   # Row 1 — chair arms + back
    [0.62,  0.95,  0.91,  0.47],   # Row 2 — chair seat (peak)
    [0.55,  0.78,  0.72,  0.42],   # Row 3 — 5-star base legs
    [0.30,  0.24,  0.28,  0.36],   # Row 4 — RF shadow
    [0.25,  0.20,  0.22,  0.32],   # Row 5 — deep shadow
], dtype=float)

GRID_ROWS, GRID_COLS = CSI_VARIANCE.shape

# ── Real-world positions (cm) ─────────────────────────────────────
X_CM = [c * CELL_W_CM for c in range(GRID_COLS)]  # 0, 27, 54, 81
Y_CM = [r * CELL_D_CM for r in range(GRID_ROWS)]  # 0, 22, 44, 66, 88, 110


def rssi_for_position(row, col):
    """Simulate RSSI: weaker far from TX (row 0), attenuated behind chair."""
    dist_factor = row / (GRID_ROWS - 1)          # 0 (near TX) → 1 (far)
    in_shadow   = 1 if row >= 4 else 0           # rows 4-5 behind chair
    rssi = BASE_RSSI - int(dist_factor * 18) - in_shadow * 7
    return max(rssi + random.randint(-3, 3), -85)


def generate_iq_packet(target_var, packet_noise=4.0):
    """
    Generate one packet of 64-subcarrier IQ data whose amplitude
    distribution reflects the given target_var (0–1 scale).

    target_var → sub-carrier spread → amplitude variance

    Physics model:
      • High variance  → strong reflections, different path lengths
                         → large spread across subcarriers
      • Low variance   → clean LOS, flat amplitude profile
    """
    # Map target_var to subcarrier spread
    # var ≈ 0.20 → spread ≈  4  (flat, clean signal)
    # var ≈ 0.95 → spread ≈ 22  (heavy scattering, large variation)
    spread = 3.0 + target_var * 20.0   # spread of subcarrier means

    # Each subcarrier has its own mean amplitude offset
    sub_means = np.random.uniform(-spread / 2, spread / 2, N_SUB)
    amp_means = np.clip(BASE_AMP + sub_means, 2.0, BASE_AMP + spread)

    iq_vals = []
    for amp in amp_means:
        noisy_amp = amp + np.random.normal(0, packet_noise)
        noisy_amp = max(noisy_amp, 1.0)
        phase = random.uniform(0, 2 * math.pi)
        I = int(round(noisy_amp * math.cos(phase)))
        Q = int(round(noisy_amp * math.sin(phase)))
        iq_vals.extend([I, Q])

    return iq_vals


def write_csv(filepath, row, col):
    """Write one CSV file for grid position (row, col)."""
    target_var = float(CSI_VARIANCE[row, col])
    x_pos      = X_CM[col]
    y_pos      = Y_CM[row]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        # Header comment — describes this measurement point
        f.write(f"# CSI data — grid [{row},{col}] at ({x_pos}cm, {y_pos}cm)\n")
        f.write(f"# Cell: {CELL_W_CM}cm wide × {CELL_D_CM}cm deep\n")
        f.write(f"# Target RF variance: {target_var:.3f}\n")
        f.write(f"# Packets: {PACKETS}  Subcarriers: {N_SUB}\n")

        ts = START_TS + (row * GRID_COLS + col) * PACKETS * PKT_INTERVAL

        for p in range(PACKETS):
            rssi    = rssi_for_position(row, col)
            extra1  = 0
            extra2  = 0
            iq      = generate_iq_packet(target_var)
            ts     += PKT_INTERVAL + random.randint(-5000, 5000)  # jitter

            iq_str = ",".join(map(str, iq))
            f.write(
                f"CSI_DATA,{ts},{rssi},{N_SUB},{extra1},{extra2},{iq_str}\n"
            )


def verify_csv(filepath):
    """Quick sanity-check: parse the CSV and report mean amplitude + variance."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from pipeline.csi_parser import parse_line

    amps = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            result = parse_line(line.strip())
            if result:
                amps.append(result["amplitude"])

    if not amps:
        return None

    mat = np.array(amps)
    mean_per_sub = np.mean(mat, axis=0)
    return float(np.var(mean_per_sub))


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("=" * 62)
    print(f"  Generating CSI CSV files  →  {OUTPUT_DIR}/")
    print(f"  Grid   : {GRID_ROWS} rows × {GRID_COLS} cols")
    print(f"  Cell   : {CELL_W_CM}cm wide × {CELL_D_CM}cm deep")
    print(f"  Packets: {PACKETS} per position")
    print("=" * 62)
    print()
    print(f"  {'File':<20} {'Pos (cm)':<18} {'Target var':>10}  {'Parsed var':>10}")
    print(f"  {'-'*20} {'-'*18} {'-'*10}  {'-'*10}")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            fname  = f"r{r:02d}_c{c:02d}.csv"
            fpath  = os.path.join(OUTPUT_DIR, fname)
            target = float(CSI_VARIANCE[r, c])

            write_csv(fpath, r, c)

            # Verify by parsing back
            parsed_var = verify_csv(fpath)
            var_str    = f"{parsed_var:.4f}" if parsed_var is not None else "N/A"

            print(f"  {fname:<20} ({X_CM[c]:3d}cm, {Y_CM[r]:3d}cm)    "
                  f"{target:>10.3f}   {var_str:>10}")

    print()
    print("=" * 62)
    print(f"  Done! {GRID_ROWS * GRID_COLS} files written to {OUTPUT_DIR}/")
    print()
    print("  View with:")
    print("    python src/chair_demo.py")
    print()
    print("  Or run full pipeline on this data:")
    print("    python src/main.py --mode visualize \\")
    print(f"      --data src/csi_data_chair --room {GRID_COLS*CELL_W_CM}x{GRID_ROWS*CELL_D_CM}x250")
    print("=" * 62)
