"""
serial_monitor.py
─────────────────────────────────────────────────────────
Standalone CSI Serial Monitor — mirrors ESP-IDF Monitor output
while printing decoded packet stats and optionally saving to CSV.

Usage:
    python src/serial_monitor.py
    python src/serial_monitor.py --port /dev/cu.usbserial-0001
    python src/serial_monitor.py --baud 115200 --save output/csi_raw.csv
    python src/serial_monitor.py --port /dev/cu.usbserial-0001 --save csi.csv --quiet

Controls:
    Ctrl+C  —  stop monitoring (CSV is safely closed first)
─────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from pipeline.serial_utils import auto_detect_port, open_serial, validate_csi_line
from pipeline.csi_parser import parse_line


# ── ANSI colour helpers ───────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _rssi_colour(rssi: int) -> str:
    if rssi >= -60:  return _GREEN
    if rssi >= -75:  return _YELLOW
    return _RED


def _fmt_bar(value: float, max_val: float = 30.0, width: int = 20) -> str:
    """Mini ASCII amplitude bar."""
    filled = int(min(value / max_val, 1.0) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ── Main monitor loop ─────────────────────────────────────────────────

def run_monitor(port: str, baud: int, save_path: str | None, quiet: bool):
    print()
    print("=" * 62)
    print(f"  {_BOLD}CSI Serial Monitor{_RESET}  —  ESP32 via serial port")
    print("=" * 62)
    print(f"  Port    : {port}")
    print(f"  Baud    : {baud}")
    print(f"  Save    : {save_path or 'disabled'}")
    print(f"  Verbose : {'off (--quiet)' if quiet else 'on'}")
    print()
    print("  Press Ctrl+C to stop.\n")

    ser = open_serial(port, baud=baud, timeout=2.0)

    csv_file   = None
    csv_writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        csv_file = open(save_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp_us", "rssi", "n_sub",
            "extra1", "extra2", "amp_mean", "amp_std",
            "zero_ratio", "raw_line"
        ])
        print(f"  [CSV] Writing to: {save_path}\n")

    n_total    = 0
    n_valid    = 0
    t_start    = time.time()
    last_stats = t_start

    try:
        while True:
            try:
                raw  = ser.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"  [read error] {e}")
                time.sleep(0.1)
                continue

            n_total += 1

            if not validate_csi_line(line):
                # Pass through non-CSI lines (e.g. ESP-IDF log messages) in verbose mode
                if not quiet and line:
                    print(f"  {_CYAN}[LOG]{_RESET} {line}")
                continue

            parsed = parse_line(line)
            if parsed is None:
                if not quiet:
                    print(f"  {_RED}[PARSE FAIL]{_RESET} {line[:80]}...")
                continue

            n_valid += 1
            amp      = parsed["amplitude"]
            amp_mean = float(np.mean(amp))
            amp_std  = float(np.std(amp))
            zero_r   = float(np.mean(amp == 0.0))

            # ── Live print ────────────────────────────────────────────
            if not quiet:
                rc = _rssi_colour(parsed["rssi"])
                bar = _fmt_bar(amp_mean)
                print(
                    f"  #{n_valid:>5}  "
                    f"ts={parsed['timestamp_us']:>14,}µs  "
                    f"RSSI={rc}{parsed['rssi']:>4}dBm{_RESET}  "
                    f"sub={parsed['n_sub']:>3}  "
                    f"amp={amp_mean:>7.2f}±{amp_std:>5.2f}  "
                    f"zero={zero_r:.1%}  {bar}"
                )

            # ── CSV write ─────────────────────────────────────────────
            if csv_writer:
                csv_writer.writerow([
                    parsed["timestamp_us"],
                    parsed["rssi"],
                    parsed["n_sub"],
                    parsed["extra1"],
                    parsed["extra2"],
                    round(amp_mean, 4),
                    round(amp_std,  4),
                    round(zero_r,   4),
                    line,
                ])

            # ── Rolling stats every 5 s ───────────────────────────────
            now = time.time()
            if now - last_stats >= 5.0:
                elapsed = now - t_start
                rate    = n_valid / max(elapsed, 0.001)
                print(
                    f"\n  {_BOLD}── Stats ──{_RESET}  "
                    f"elapsed={elapsed:.0f}s  "
                    f"total={n_total}  valid={n_valid}  "
                    f"rate={rate:.1f} pkt/s\n"
                )
                last_stats = now

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        rate    = n_valid / max(elapsed, 0.001)
        print("\n")
        print("=" * 62)
        print("  Monitoring stopped.")
        print(f"  Duration   : {elapsed:.1f}s")
        print(f"  Lines read : {n_total}")
        print(f"  CSI valid  : {n_valid}  ({rate:.1f} pkt/s)")
        if save_path:
            print(f"  CSV saved  : {save_path}")
        print("=" * 62)

    finally:
        ser.close()
        if csv_file:
            csv_file.flush()
            csv_file.close()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="serial_monitor.py",
        description="ESP32 CSI Serial Monitor — streams and decodes CSI_DATA lines",
    )
    parser.add_argument("--port",  default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",  type=int, default=115200,
                        help="Baud rate (default: 115200 — matches ESP-IDF Monitor)")
    parser.add_argument("--save",  default=None, metavar="FILE",
                        help="Save decoded CSI to this CSV file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-packet output (only show rolling stats)")

    args = parser.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        print("[serial_monitor] No serial port found. Plug in the ESP32 and retry.")
        print("  Or specify manually: --port /dev/cu.usbserial-XXXX")
        sys.exit(1)

    run_monitor(port=port, baud=args.baud, save_path=args.save, quiet=args.quiet)


if __name__ == "__main__":
    main()
