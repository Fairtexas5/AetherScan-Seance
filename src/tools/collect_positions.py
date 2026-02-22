"""
collect_positions.py — Phase S-5 Guided Position Collector (macOS)
─────────────────────────────────────────────────────────
Guided walk-around capture using Wireshark/tshark via pyshark.
Prompts you to move to each grid position, captures WiFi RSSI
for 8 seconds per position, then saves a session .pkl file.

Usage:
    sudo python tools/collect_positions.py \\
        --rows 4 --cols 5 --room 5x4x2.5 --interface en0

    # Then visualize:
    python main.py --mode load --file room_capture.pkl

Requirements:
    - Wireshark / tshark installed  (brew install --cask wireshark)
    - pyshark installed             (pip install pyshark)
    - Run with sudo for monitor mode
─────────────────────────────────────────────────────────
"""

import os
import sys
import pickle
import argparse
import time
import numpy as np

# ── Ensure project root on path ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_room(room_str: str) -> tuple:
    parts = room_str.lower().replace(" ", "").split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid room format '{room_str}'. Use WxDxH e.g. 5x4x2.5")
    return tuple(float(p) for p in parts)


def run_guided_capture(
    rows: int,
    cols: int,
    room_w: float,
    room_d: float,
    room_h: float,
    interface: str,
    capture_s: float,
    output_file: str,
):
    from pipeline.capture_macos import capture_wifi_frames, print_rssi_summary

    total_positions = rows * cols
    pos_num = 0

    print()
    print("=" * 62)
    print("  WiFi CSI Guided Position Collector")
    print("=" * 62)
    print(f"  Grid      : {rows} rows × {cols} cols = {total_positions} positions")
    print(f"  Room      : {room_w}m × {room_d}m × {room_h}m")
    print(f"  Interface : {interface}")
    print(f"  Capture/pt: {capture_s}s")
    print(f"  Output    : {output_file}")
    print()
    print("  Instructions:")
    print("  1. Keep the transmitter (NodeMCU/router) in one fixed corner.")
    print("  2. Move to each position when prompted.")
    print("  3. Stand 0.5m from the receiver while it captures.")
    print("  4. Don't move during the countdown.")
    print()
    input("  Press Enter to begin... ")

    session_points = []
    voxel_rows = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            pos_num += 1
            x_m = (c / max(cols - 1, 1)) * room_w
            y_m = (r / max(rows - 1, 1)) * room_d
            z_m = 1.0  # receiver height

            print(f"\n[POS {r+1}/{rows}, {c+1}/{cols}]")
            print(f"  Move to: x={x_m:.2f}m from left wall, "
                  f"y={y_m:.2f}m from front wall")
            input(f"  Press Enter when ready to capture for {capture_s:.0f}s...")

            # Countdown
            for i in range(3, 0, -1):
                print(f"  Capturing in {i}...", end="\r")
                time.sleep(1)
            print("  Capturing...       ")

            frames = capture_wifi_frames(
                interface=interface,
                duration_s=capture_s,
            )

            if not frames:
                print("  [WARN] No frames captured at this position — using 0.")
                avg_rssi = 0.0
            else:
                rssi_vals = [rv for rv, _, _ in frames]  # rv avoids shadowing loop var r
                avg_rssi = float(np.mean(rssi_vals))
                print(f"  ✓ {len(frames)} frames, avg RSSI = {avg_rssi:.1f} dBm")

            voxel_rows[r, c] = avg_rssi
            session_points.append({
                "row": r, "col": c,
                "x_m": round(x_m, 3),
                "y_m": round(y_m, 3),
                "z_m": round(z_m, 3),
                "n_frames": len(frames),
                "avg_rssi": avg_rssi,
                "frames": frames,
            })

    print("\n  ✓ All positions collected!")

    # Build 3D voxel grid
    g = voxel_rows.copy()
    finite = g[np.isfinite(g) & (g != 0)]
    if len(finite) > 0:
        gmin, gmax = finite.min(), finite.max()
        g = np.clip((g - gmin) / (gmax - gmin + 1e-9), 0, 1)
    nH = max(1, int(round(room_h / (room_d / max(rows, 1)))))
    voxel_grid_3d = np.stack([g] * nH, axis=-1)

    session = {
        "source": "wireshark_guided",
        "room_w": room_w,
        "room_d": room_d,
        "room_h": room_h,
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_points": session_points,
        "voxel_grid_3d": voxel_grid_3d,
        "voxel_feature_grid_2d": voxel_rows,
        "total_frames": sum(len(p["frames"]) for p in session_points),
    }

    with open(output_file, "wb") as f:
        pickle.dump(session, f)

    print(f"\n[DONE] Session saved to: {output_file}")
    print(f"  Visualize with:  python main.py --mode load --file {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Guided per-position WiFi capture via Wireshark/tshark (macOS)."
    )
    parser.add_argument("--rows",      type=int,   required=True, help="Grid rows")
    parser.add_argument("--cols",      type=int,   required=True, help="Grid cols")
    parser.add_argument("--room",      required=True, help="WxDxH e.g. 5x4x2.5")
    parser.add_argument("--interface", default="en0", help="WiFi interface (default: en0)")
    parser.add_argument("--duration",  type=float, default=8.0,
                        help="Capture duration per position in seconds (default: 8)")
    parser.add_argument("--output",    default="room_capture.pkl", help="Output .pkl file")
    args = parser.parse_args()

    room_w, room_d, room_h = parse_room(args.room)

    run_guided_capture(
        rows=args.rows,
        cols=args.cols,
        room_w=room_w,
        room_d=room_d,
        room_h=room_h,
        interface=args.interface,
        capture_s=args.duration,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
