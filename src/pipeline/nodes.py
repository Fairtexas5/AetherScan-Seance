"""
nodes.py
LangGraph node functions for the CSI pipeline.
Each node receives state, does one job, returns updated state.
No LLM / No API key — LangGraph used purely as a state machine.
"""

import os
import time
import csv
from typing import Any

from pipeline.serial_utils import auto_detect_port, open_serial, read_csi_line, flush_stale_data
from pipeline.validator import validate_grid_point, print_validation_report
from pipeline import heatmap as hm


# ─────────────────────────────────────────────
# NODE 1: Setup
# ─────────────────────────────────────────────
def node_setup(state: dict) -> dict:
    """
    Auto-detect serial port, create output directories, print run summary.
    """
    print("\n" + "="*60)
    print("  CSI Room Mapper — Automated Pipeline (No API Key)")
    print("="*60)

    # Auto-detect port if not manually set
    port = state.get("serial_port") or auto_detect_port()
    if not port:
        raise RuntimeError("No serial port detected. Plug in your ESP32 and retry.")

    # Create directories
    os.makedirs(state["data_folder"], exist_ok=True)
    os.makedirs(state["output_folder"], exist_ok=True)

    total_points = state["grid_rows"] * state["grid_cols"]
    print(f"\n  Grid         : {state['grid_rows']} rows × {state['grid_cols']} cols = {total_points} points")
    print(f"  Room size    : {state['room_width_m']*100:.0f}cm × {state['room_height_m']*100:.0f}cm")
    print(f"  Packets/pt   : {state['packets_per_point']}")
    print(f"  Serial port  : {port}")
    print(f"  Data folder  : {state['data_folder']}")
    print(f"  Output folder: {state['output_folder']}")
    print(f"  Live preview : {'ON' if state.get('live_preview') else 'OFF'}")
    print(f"  Auto-retry   : {'ON' if state.get('auto_retry') else 'OFF'}")
    print()

    return {**state, "serial_port": port, "setup_done": True, "current_row": 0, "current_col": 0}


# ─────────────────────────────────────────────
# NODE 2: Collect one grid point
# ─────────────────────────────────────────────
def node_collect_point(state: dict) -> dict:
    """
    Collect CSI packets for (current_row, current_col).
    Prompts user to position the ESP32, then auto-collects.
    Saves raw CSV. Updates state with collected packet list.
    """
    row = state["current_row"]
    col = state["current_col"]
    total = state["grid_rows"] * state["grid_cols"]
    done = row * state["grid_cols"] + col + 1

    print(f"\n──────────────────────────────────────")
    print(f"  Point {done}/{total}  →  row={row}, col={col}")
    print(f"  Place ESP32 at grid position (row={row}, col={col})")
    input("  Press Enter when ESP32 is in position and steady... ")

    ser = open_serial(state["serial_port"])
    flush_stale_data(ser, flush_seconds=0.5)

    packets = []
    target = state["packets_per_point"]
    print(f"  Collecting {target} packets", end="", flush=True)

    start = time.time()
    last_report = start
    try:
        while len(packets) < target:
            line = read_csi_line(ser)
            if line:
                packets.append(line)
                now = time.time()
                if now - last_report >= 2.0:   # update every 2 s
                    elapsed = now - start
                    rate    = len(packets) / max(elapsed, 0.001)
                    remain  = (target - len(packets)) / max(rate, 0.001)
                    print(
                        f"\r  Collecting {target} packets  "
                        f"[{len(packets)}/{target}]  "
                        f"{rate:.0f} pkt/s  "
                        f"ETA {remain:.0f}s   ",
                        end="", flush=True
                    )
                    last_report = now
    finally:
        ser.close()

    elapsed = time.time() - start
    rate    = len(packets) / max(elapsed, 0.001)
    print(f"\r  Collected {len(packets)} packets in {elapsed:.1f}s  ({rate:.0f} pkt/s)          ")

    # Save to CSV
    fname = os.path.join(state["data_folder"], f"r{row:02d}_c{col:02d}.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CSI_DATA", "timestamp", "rssi", "n_subcarriers", "iq_values..."])
        for pkt in packets:
            writer.writerow(pkt.split(","))

    return {**state, "last_packets": packets, "last_point": (row, col), "collection_done": False}


# ─────────────────────────────────────────────
# NODE 3 (LEGACY — NOT WIRED INTO GRAPH)
# The active pipeline uses node_ai_validate_point
# from pipeline/ai_validator.py instead.
# Kept here for reference / fallback use only.
# ─────────────────────────────────────────────
def node_validate_point(state: dict) -> dict:
    """
    [LEGACY] Rule-based quality check — superseded by node_ai_validate_point.
    NOT used in the current LangGraph pipeline (see graph.py).
    Run quality checks on the last collected grid point.
    Sets 'last_valid' flag. If invalid and auto_retry=True, flags for recollection.
    """
    row, col = state["last_point"]
    packets = state["last_packets"]

    is_valid, reason = validate_grid_point(packets)
    print_validation_report(row, col, is_valid, reason)

    if not is_valid and state.get("auto_retry", True):
        retry_count = state.get("retry_count", 0) + 1
        MAX_RETRIES = 3
        if retry_count <= MAX_RETRIES:
            print(f"  ↻ Auto-retry {retry_count}/{MAX_RETRIES} for r={row} c={col}")
            return {**state, "last_valid": False, "retry_count": retry_count}
        else:
            print(f"  ⚠ Max retries reached for r={row} c={col}. Accepting data as-is.")

    return {**state, "last_valid": True, "retry_count": 0}


# ─────────────────────────────────────────────
# NODE 4: Live preview (optional)
# ─────────────────────────────────────────────
def node_live_preview(state: dict) -> dict:
    """
    Regenerate partial heatmap after each successful grid point.
    Only runs if state['live_preview'] is True.
    """
    if not state.get("live_preview", False):
        return state

    row, col = state["last_point"]
    label = f"r{row}c{col}"
    print(f"  [preview] Regenerating heatmap...")

    hm.live_preview(
        data_folder=state["data_folder"],
        grid_rows=state["grid_rows"],
        grid_cols=state["grid_cols"],
        room_width_m=state["room_width_m"],
        room_height_m=state["room_height_m"],
        output_folder=state["output_folder"],
        label=label
    )
    return state


# ─────────────────────────────────────────────
# NODE 5: Advance to next grid point
# ─────────────────────────────────────────────
def node_advance(state: dict) -> dict:
    """
    Move current_row/current_col to the next grid position.
    Sets collection_done=True when all points are collected.
    """
    row = state["current_row"]
    col = state["current_col"] + 1

    if col >= state["grid_cols"]:
        col = 0
        row += 1

    if row >= state["grid_rows"]:
        print("\n  ✓ All grid points collected!")
        return {**state, "collection_done": True, "current_row": row, "current_col": col}

    return {**state, "current_row": row, "current_col": col, "collection_done": False}


# ─────────────────────────────────────────────
# NODE 6: Process final heatmap
# ─────────────────────────────────────────────
def node_process_heatmap(state: dict) -> dict:
    """
    Process all collected CSI data and generate the final heatmap PNG.
    """
    print("\n" + "="*60)
    print("  Processing all CSI data → Final Heatmap")
    print("="*60)

    output_path = hm.final_heatmap(
        data_folder=state["data_folder"],
        grid_rows=state["grid_rows"],
        grid_cols=state["grid_cols"],
        room_width_m=state["room_width_m"],
        room_height_m=state["room_height_m"],
        output_folder=state["output_folder"]
    )

    print(f"\n  ✓ Final heatmap saved: {output_path}")
    return {**state, "heatmap_path": output_path, "pipeline_done": True}
