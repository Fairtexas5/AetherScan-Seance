"""
csi_parser.py
─────────────────────────────────────────────────────────
Single source-of-truth for parsing CSI_DATA lines emitted by
app_main.c over UART / ESP-IDF Monitor.

Line format (app_main.c csi_callback):
  CSI_DATA,{timestamp_us},{rssi},{n_sub},{extra1},{extra2},{I0},{Q0},{I1},{Q1},...

Field indices:
  0  → "CSI_DATA"      literal tag
  1  → timestamp_us    microseconds since boot (uint64)
  2  → rssi            received signal strength (dBm, int8)
  3  → n_sub           number of subcarriers (== len/2 of raw buf)
  4  → extra1          additional metadata byte from firmware
  5  → extra2          additional metadata byte from firmware
  6+ → IQ values       alternating I (even idx) and Q (odd idx)

The IQ start index is 6 — this is the fix for the off-by-2 bug
that was extracting metadata fields into the IQ array.
─────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────
CSI_TAG       = "CSI_DATA"
CSI_IQ_START  = 6          # field index where IQ data begins
MIN_IQ_VALUES = 4          # must have at least 2 I/Q pairs


# ── Core parser ──────────────────────────────────────────────────────

def parse_line(line: str) -> Optional[dict]:
    """
    Parse one raw CSI_DATA line string.

    Returns a dict with:
        timestamp_us : int
        rssi         : int   (dBm)
        n_sub        : int   (number of subcarriers)
        extra1       : int
        extra2       : int
        iq_raw       : list[int]   raw interleaved [I0, Q0, I1, Q1, ...]
        I            : np.ndarray  in-phase values
        Q            : np.ndarray  quadrature values
        amplitude    : np.ndarray  sqrt(I^2 + Q^2) per subcarrier

    Returns None if the line is malformed, missing fields, or non-numeric.
    """
    if not line:
        return None

    parts = line.strip().split(",")

    # Must start with CSI_DATA and have at least header + 1 IQ pair
    if len(parts) < CSI_IQ_START + MIN_IQ_VALUES:
        return None
    if parts[0].strip() != CSI_TAG:
        return None

    try:
        timestamp_us = int(parts[1].strip())
        rssi         = int(parts[2].strip())
        n_sub        = int(parts[3].strip())
        extra1       = int(parts[4].strip())
        extra2       = int(parts[5].strip())
    except (ValueError, IndexError):
        return None

    # Parse IQ values
    try:
        iq_raw = [int(v.strip()) for v in parts[CSI_IQ_START:] if v.strip() != ""]
    except ValueError:
        return None

    if len(iq_raw) < MIN_IQ_VALUES:
        return None

    # Split interleaved IQ → separate I and Q arrays
    I_arr = np.array(iq_raw[0::2], dtype=np.float32)
    Q_arr = np.array(iq_raw[1::2], dtype=np.float32)
    amp   = np.sqrt(I_arr ** 2 + Q_arr ** 2)

    return {
        "timestamp_us": timestamp_us,
        "rssi":         rssi,
        "n_sub":        n_sub,
        "extra1":       extra1,
        "extra2":       extra2,
        "iq_raw":       iq_raw,
        "I":            I_arr,
        "Q":            Q_arr,
        "amplitude":    amp,
    }


def parse_iq(line: str) -> Optional[np.ndarray]:
    """
    Compatibility shim — returns amplitude array only, or None on failure.
    Drop-in replacement for the inline parse_iq() that existed in validator.py.
    """
    result = parse_line(line)
    if result is None:
        return None
    return result["amplitude"]


def zero_subcarrier_ratio(amp: np.ndarray) -> float:
    """Return fraction of subcarriers that are exactly zero (DC nulls etc.)."""
    if amp.size == 0:
        return 1.0
    return float(np.mean(amp == 0.0))


# ── Quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Take the very first line from sample.txt for a deterministic check
    SAMPLE = (
        "CSI_DATA,23503655838,-50,64,45,-47,18,0,"
        "-13,13,-12,13,-12,13,-12,13,-11,12,-11,12,-11,13,-10,13,-10,13,-10,13,"
        "-9,13,-9,12,-9,12,-9,12,-9,12,-9,11,-9,11,-9,11,-9,10,-9,10,-8,9,-8,8,"
        "-8,8,-8,8,-8,7,-2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,2,"
        "-7,8,-7,7,-8,8,-8,8,-9,8,-9,9,-10,9,-10,9,-11,9,-11,9,-12,9,-12,9,"
        "-12,9,-12,9,-13,10,-13,10,-13,10,-13,11,-13,11,-13,11,-13,11,-13,12,"
        "-13,12,-12,12,-13,12,-13,13"
    )

    r = parse_line(SAMPLE)
    assert r is not None,            "parse_line returned None on valid sample"
    assert r["rssi"]  == -50,        f"RSSI mismatch: {r['rssi']}"
    assert r["n_sub"] == 64,         f"n_sub mismatch: {r['n_sub']}"
    assert r["I"][0]  == 18.0,       f"I[0] should be 18, got {r['I'][0]}"
    assert r["Q"][0]  == 0.0,        f"Q[0] should be 0,  got {r['Q'][0]}"
    assert r["amplitude"].shape[0] == len(r["I"]), "amplitude/I length mismatch"

    print("✓ csi_parser self-test passed")
    print(f"  timestamp : {r['timestamp_us']:,} µs")
    print(f"  rssi      : {r['rssi']} dBm")
    print(f"  n_sub     : {r['n_sub']}")
    print(f"  IQ pairs  : {len(r['I'])}")
    print(f"  amp[0:4]  : {r['amplitude'][:4]}")
    print(f"  zero ratio: {zero_subcarrier_ratio(r['amplitude']):.1%}")
