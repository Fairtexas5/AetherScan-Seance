"""
main.py
─────────────────────────────────────────────────────────
CSI Room Mapper — Multi-Mode Entry Point
Hardware: MacBook Pro (macOS)
AI Model: phi3.5:latest via Ollama (fully local, no API key)

MODES
─────
  python main.py --mode simulate [--room 5x4x2.5]
      Generate synthetic 3D CSI data and launch Plotly viewer.
      No hardware needed. Good first run.

  sudo python main.py --mode capture --interface en0 [--duration 60]
      Passive WiFi capture via Wireshark / tshark (pyshark).
      Requires: brew install --cask wireshark && pip install pyshark
      Requires: sudo for monitor mode access.

  python main.py --mode load --file room_capture.pkl
      Load a saved .pkl session and relaunch the 3D viewer.

  python main.py [no args]
      Run the full AI-enhanced LangGraph live-collection pipeline
      (requires ESP32 plugged in via USB and Ollama running).

ONE-TIME SETUP:
    pip install -r requirements.txt
    brew install --cask wireshark   # for capture mode
    ollama pull phi3.5:latest        # for AI pipeline mode
    ollama serve                     # start AI backend

MAC NOTES:
    • WiFi interface is usually 'en0' (not 'Wi-Fi' as on Windows)
    • Capture mode needs sudo for monitor mode
    • Npcap is Windows-only — we use pyshark/tshark on macOS
─────────────────────────────────────────────────────────
"""

import os
import sys
import argparse

# ── Ensure project root on sys.path ─────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════
#  Parse room dimensions helper
# ═══════════════════════════════════════════════════════

def _parse_room(room_str: str) -> tuple:
    """'500x400x250' cm → (5.0, 4.0, 2.5) m internally"""
    try:
        parts = room_str.lower().replace(" ", "").split("x")
        if len(parts) != 3:
            raise ValueError
        return tuple(float(p) / 100.0 for p in parts)   # cm → m
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid room format '{room_str}'. Expected WxDxH in cm e.g. 500x400x250"
        )


# ═══════════════════════════════════════════════════════
#  SIMULATE mode
# ═══════════════════════════════════════════════════════

def run_simulate(room_w, room_d, room_h, grid_res, output_folder):
    from pipeline.simulator import simulate_room
    from pipeline.visualizer_3d import launch_3d_viewer

    print()
    print("=" * 60)
    print("  WiFi CSI 3D Room Mapper")
    print("=" * 60)
    nW = max(1, int(round(room_w / grid_res)))
    nD = max(1, int(round(room_d / grid_res)))
    nH = max(1, int(round(room_h / grid_res)))
    print(f"  Mode  : SIMULATION")
    print(f"  Room  : {room_w*100:.0f}cm x {room_d*100:.0f}cm x {room_h*100:.0f}cm")
    print(f"  Grid  : {nW}x{nD}x{nH} voxels  ({grid_res*100:.0f}cm resolution)")
    print("=" * 60)
    print()

    print("[PIPELINE] Generating synthetic CSI data...")
    voxel_grid, frames, shape = simulate_room(
        room_w=room_w, room_d=room_d, room_h=room_h,
        grid_res=grid_res, verbose=True
    )
    print(f"\n[PIPELINE] Step 1/3 - Signal Processing... (done)")
    print(f"[PIPELINE] Step 2/3 - 3D Room Reconstruction...")
    print(f"  Volume shape: {shape}")
    print()
    print("[PIPELINE] Step 3/3 - Launching 3D Animated Slice Viewer...")

    os.makedirs(output_folder, exist_ok=True)
    html_path = launch_3d_viewer(
        voxel_grid,
        room_w=room_w, room_d=room_d, room_h=room_h,
        output_folder=output_folder,
        title="WiFi CSI 3D Room Map — Simulation",
    )

    print()
    print("=" * 60)
    print("  Simulation complete!")
    print(f"  3D Viewer → {html_path}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
#  CAPTURE mode (macOS / Wireshark / pyshark)
# ═══════════════════════════════════════════════════════

def run_capture(interface, duration, room_w, room_d, room_h, output_folder):
    from pipeline.capture_macos import (
        capture_wifi_frames, print_rssi_summary, frames_to_rssi_grid
    )
    from pipeline.visualizer_3d import launch_3d_viewer

    print()
    print("=" * 60)
    print("  WiFi CSI 3D Room Mapper — Capture Mode (macOS)")
    print("=" * 60)
    print(f"  Interface : {interface}")
    print(f"  Duration  : {duration}s")
    print(f"  Room      : {room_w*100:.0f}cm x {room_d*100:.0f}cm x {room_h*100:.0f}cm")
    print(f"  Backend   : Wireshark / tshark / pyshark")
    print("=" * 60)
    print()
    print("  Note: This captures 802.11 beacon RSSI (signal strength).")
    print("  For true CSI (IQ values) use the ESP32 hardware path.")
    print()

    frames = capture_wifi_frames(
        interface=interface,
        duration_s=duration,
    )
    print_rssi_summary(frames)

    if not frames:
        print("[CAPTURE] No frames captured. Check interface name and sudo permissions.")
        print(f"  Try:  sudo python main.py --mode capture --interface en0 --duration 30")
        return

    print("[PIPELINE] Building 3D voxel map from RSSI data...")
    voxel_grid = frames_to_rssi_grid(
        frames, room_w=room_w, room_d=room_d, room_h=room_h
    )

    print("[PIPELINE] Launching 3D viewer...")
    os.makedirs(output_folder, exist_ok=True)
    html_path = launch_3d_viewer(
        voxel_grid,
        room_w=room_w, room_d=room_d, room_h=room_h,
        output_folder=output_folder,
        title=f"WiFi CSI 3D Room Map — Live Capture ({interface})",
    )

    print()
    print("=" * 60)
    print(f"  3D Viewer → {html_path}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
#  LOAD mode
# ═══════════════════════════════════════════════════════

def run_load(pkl_file, output_folder):
    import pickle
    from pipeline.visualizer_3d import launch_3d_viewer

    if not os.path.exists(pkl_file):
        print(f"[LOAD] File not found: '{pkl_file}'")
        sys.exit(1)

    print(f"[LOAD] Loading session: {pkl_file}")
    with open(pkl_file, "rb") as f:
        session = pickle.load(f)

    source    = session.get("source", "unknown")
    room_w    = session.get("room_w", 5.0)
    room_d    = session.get("room_d", 4.0)
    room_h    = session.get("room_h", 2.5)
    voxel_grid = session.get("voxel_grid_3d")
    n_frames  = session.get("total_frames", 0)

    print(f"  Source     : {source}")
    print(f"  Room       : {room_w}m x {room_d}m x {room_h}m")
    print(f"  Voxel grid : {voxel_grid.shape if voxel_grid is not None else 'N/A'}")
    print(f"  Frames     : {n_frames:,}")
    print()

    if voxel_grid is None:
        print("[LOAD] Session has no voxel_grid_3d. Please re-run the converter.")
        sys.exit(1)

    print("[LOAD] Launching 3D viewer...")
    os.makedirs(output_folder, exist_ok=True)
    html_path = launch_3d_viewer(
        voxel_grid,
        room_w=room_w, room_d=room_d, room_h=room_h,
        output_folder=output_folder,
        title=f"WiFi CSI 3D Room Map — {source}",
    )

    print(f"\n  3D Viewer → {html_path}")


# ═══════════════════════════════════════════════════════
#  VISUALIZE mode (Plotly charts from saved csi_data/)
# ═══════════════════════════════════════════════════════

def run_visualize(data_folder, room_w, room_d, room_h, output_folder):
    import glob, re
    from pipeline.heatmap import build_grid
    from pipeline.csi_visualizer import build_all_charts

    print()
    print("=" * 60)
    print("  CSI Plotly Visualizer")
    print("=" * 60)
    print(f"  Data folder : {data_folder}")
    print(f"  Room        : {room_w*100:.0f}cm × {room_d*100:.0f}cm × {room_h*100:.0f}cm")
    print()

    files = glob.glob(os.path.join(data_folder, "r??_c??.csv"))
    if not files:
        print(f"[visualize] No CSV files found in '{data_folder}'.")
        print("  Run the AI pipeline first, or specify a different --data folder.")
        return

    max_r = max_c = 0
    for f in files:
        m = re.search(r"r(\d+)_c(\d+)\.csv", os.path.basename(f))
        if m:
            max_r = max(max_r, int(m.group(1)))
            max_c = max(max_c, int(m.group(2)))
    grid_rows, grid_cols = max_r + 1, max_c + 1
    print(f"  Auto-detected grid: {grid_rows} rows × {grid_cols} cols")
    print(f"  CSV files found  : {len(files)}")
    print()

    grid2d = build_grid(data_folder, grid_rows, grid_cols)
    os.makedirs(output_folder, exist_ok=True)

    paths = build_all_charts(grid2d, room_w, room_d, room_h, output_folder)

    print()
    print("=" * 60)
    print("  Charts saved — open in any browser:")
    for name, path in paths.items():
        print(f"    [{name:12s}]  {path}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
#  AI PIPELINE mode (original LangGraph ESP32 pipeline)
# ═══════════════════════════════════════════════════════

# ── LangGraph / AI Pipeline config ──────────────────────────────────
CONFIG = {
    "serial_port":     None,   # None = auto-detect ESP32
    "grid_rows":       8,
    "grid_cols":       6,
    "room_width_m":    3.0,
    "room_height_m":   4.0,
    "packets_per_point": 200,
    "live_preview":    True,
    "auto_retry":      True,
    "data_folder":     "csi_data",
    "output_folder":   "output",
    "setup_done":      False,
    "current_row":     0,
    "current_col":     0,
    "last_packets":    [],
    "last_point":      None,
    "last_valid":      True,
    "retry_count":     0,
    "collection_done": False,
    "heatmap_path":    None,
    "pipeline_done":   False,
    "anomaly_log":     [],
    "ai_report_path":  None,
    "ai_interpretation": None,
}


def run_ai_pipeline():
    from pipeline.graph import build_pipeline

    print("\n" + "=" * 60)
    print("  CSI Room Mapper — AI Pipeline (phi3.5:latest / Ollama)")
    print("=" * 60)
    print()
    print("  Pre-flight checklist:")
    print("    ✓ ESP32 plugged into laptop via USB")
    print("    ✓ NodeMCU/ESP-12F powered from power bank")
    print("    ✓ Ollama running in background  (ollama serve)")
    print("    ✓ phi3.5:latest downloaded      (ollama pull phi3.5:latest)")
    print()
    print("  Expected LLM call times on MacBook Pro:")
    print("    Validation   : ~5-15s per grid point")
    print("    Anomaly check: ~10-20s per grid point")
    print("    Final report : ~20-40s (once at end)")
    print()

    pipeline = build_pipeline()
    final_state = pipeline.invoke(CONFIG)

    print("\n" + "=" * 60)
    print("  All done!")
    print(f"  Heatmap   → {final_state.get('heatmap_path', 'output/room_heatmap_final.png')}")
    print(f"  AI Report → {final_state.get('ai_report_path', 'output/ai_report.txt')}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="WiFi CSI 3D Room Mapper — macOS (Wireshark/tshark backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode simulate --room 5x4x2.5
  sudo python main.py --mode capture --interface en0 --duration 60
  python main.py --mode load --file esp32_session.pkl
  python main.py                              # AI pipeline (needs ESP32 + Ollama)
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["simulate", "capture", "load", "visualize"],
        default=None,
        help="Run mode. Omit for AI LangGraph pipeline.",
    )
    parser.add_argument(
        "--room",
        default="500x400x250",
        help="Room dimensions WxDxH in centimetres (default: 500x400x250 = 5m x 4m x 2.5m)",
    )
    parser.add_argument(
        "--interface",
        default="en0",
        help="WiFi interface name for capture mode (default: en0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Capture duration in seconds for capture mode (default: 60)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to .pkl session file for load mode",
    )
    parser.add_argument(
        "--grid-res",
        type=float,
        default=25.0,
        dest="grid_res",
        help="Voxel resolution in centimetres for simulate mode (default: 25 = 0.25m)",
    )
    parser.add_argument(
        "--data",
        default="csi_data",
        help="CSI data folder for visualize mode (default: csi_data)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output folder for HTML viewer and PNGs (default: output)",
    )

    args = parser.parse_args()

    # Parse room dimensions
    room_w, room_d, room_h = _parse_room(args.room)
    grid_res_m = args.grid_res / 100.0  # cm → m

    if args.mode == "simulate":
        run_simulate(room_w, room_d, room_h, grid_res_m, args.output)

    elif args.mode == "capture":
        run_capture(args.interface, args.duration, room_w, room_d, room_h, args.output)

    elif args.mode == "load":
        if not args.file:
            parser.error("--mode load requires --file <path.pkl>")
        run_load(args.file, args.output)

    elif args.mode == "visualize":
        run_visualize(args.data, room_w, room_d, room_h, args.output)

    else:
        # Default: run the full AI LangGraph pipeline (original behaviour)
        run_ai_pipeline()


if __name__ == "__main__":
    main()
