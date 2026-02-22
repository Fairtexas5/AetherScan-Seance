"""
validator.py
Validates CSI data quality for each grid point BEFORE moving on.
Catches bad data early — corrupted packets, all-zero IQ, too few packets.
No API key required.
"""

import numpy as np
from typing import List, Tuple

from pipeline.csi_parser import parse_line, parse_iq

# Thresholds — tune these based on your hardware
MIN_PACKETS = 50           # Minimum acceptable packets per grid point
MAX_ZERO_RATIO = 0.5       # If >50% of IQ values are zero, data is bad
MIN_VARIANCE = 0.1         # CSI variance below this = probably no signal
MAX_VARIANCE = 1e8         # Unrealistically high = noise/glitch


def validate_grid_point(packets: List[str]) -> Tuple[bool, str]:
    """
    Validate a list of raw CSI_DATA line strings for one grid point.

    Returns:
        (is_valid: bool, reason: str)
    """
    if len(packets) < MIN_PACKETS:
        return False, f"Too few packets: {len(packets)} < {MIN_PACKETS}"

    amplitudes = []
    for line in packets:
        amp = parse_iq(line)
        if amp is not None:
            amplitudes.append(amp)

    if not amplitudes:
        return False, "No parseable CSI packets found."

    stacked = np.vstack(amplitudes)

    # Check zero ratio
    zero_ratio = np.mean(stacked == 0)
    if zero_ratio > MAX_ZERO_RATIO:
        return False, f"Too many zero IQ values: {zero_ratio:.1%} zeros"

    # Check variance (signal health)
    mean_amp = np.mean(stacked, axis=0)
    variance = float(np.var(mean_amp))
    if variance < MIN_VARIANCE:
        return False, f"Signal variance too low ({variance:.4f}) — check WiFi connection"
    if variance > MAX_VARIANCE:
        return False, f"Signal variance unrealistically high ({variance:.2e}) — possible noise"

    # Check RSSI from packet headers
    rssi_values = []
    for line in packets:
        parsed = parse_line(line)
        if parsed is not None:
            rssi_values.append(parsed["rssi"])
    if rssi_values:
        avg_rssi = np.mean(rssi_values)
        if avg_rssi < -90:
            return False, f"Very weak signal: avg RSSI = {avg_rssi:.1f} dBm"

    return True, f"OK — {len(amplitudes)} valid packets, variance={variance:.4f}"


def print_validation_report(row: int, col: int, is_valid: bool, reason: str):
    status = "✓ PASS" if is_valid else "✗ FAIL"
    color = "\033[92m" if is_valid else "\033[91m"
    reset = "\033[0m"
    print(f"  [{color}{status}{reset}] r={row} c={col} → {reason}")
