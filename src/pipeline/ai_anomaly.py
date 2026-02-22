"""
ai_anomaly.py
─────────────────────────────────────────────────────────
LLM-based anomaly detection for CSI signals.
Tuned for MacBook Pro / phi3.5:latest via Ollama.

Approach — pure LLM (no ML library):
    1. Compute lightweight statistics from current + all past grid points
       (mean amplitude, variance, RSSI, zero ratio, subcarrier deviation)
    2. Format a compact comparison table — concise for phi3.5:latest
    3. Ask the LLM: "Is this point unusual? What might cause it?"

Why pure LLM instead of Isolation Forest on your hardware:
    - Isolation Forest on CPU with 48+ points is fine actually, BUT
      the user specifically wants LLM-based explanations
    - phi3.5:latest handles statistical comparison tables well
    - Keeps the dependency list smaller (no scikit-learn needed)

Where it fits:
    [ai_validate_point] → [ai_anomaly_detection] → [live_preview]
─────────────────────────────────────────────────────────
"""

import os
import numpy as np
from typing import List
from pipeline.ai_client import ask_llm
from pipeline.csi_parser import parse_line


# ── Keep prompts concise for faster inference ──
ANOMALY_SYSTEM_PROMPT = """
You are a WiFi signal expert helping analyze CSI (Channel State Information)
room mapping data from an ESP32 experiment.

You will get statistics for a new grid point compared to all previous points.
Decide if this point is NORMAL or ANOMALOUS, and give a brief physical explanation.

Respond in this EXACT format (3 lines only):
VERDICT: NORMAL   or   VERDICT: ANOMALOUS
CONFIDENCE: LOW / MEDIUM / HIGH
REASON: <one sentence — what might cause this physically in a room>
""".strip()


def _parse_amps_from_csv(filepath: str) -> np.ndarray | None:
    """Load one CSV file, return mean amplitude vector across all packets."""
    amps = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header row
            result = parse_line(line.strip())
            if result is None:
                continue
            amps.append(result["amplitude"])
    except Exception:
        return None
    if not amps:
        return None
    return np.mean(np.vstack(amps), axis=0)


def _parse_rssi_from_packets(packets: List[str]) -> list:
    rssi_vals = []
    for line in packets:
        try:
            parts = line.strip().split(",")
            rssi_vals.append(int(parts[2]))
        except Exception:
            pass
    return rssi_vals


def _parse_amps_from_packets(packets: List[str]) -> np.ndarray | None:
    """Parse current point packets (in-memory, not from file yet)."""
    amps = []
    for line in packets:
        result = parse_line(line)
        if result is None:
            continue
        amps.append(result["amplitude"])
    if not amps:
        return None
    return np.mean(np.vstack(amps), axis=0)


def _load_baseline_stats(data_folder: str, exclude: tuple) -> dict | None:
    """
    Load all collected CSVs (except current point) and compute baseline stats.
    Returns dict with mean, std, variance stats across the room so far.
    """
    all_means = []
    all_vars  = []
    all_rssis = []

    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith(".csv"):
            continue
        try:
            base = fname.replace(".csv", "").split("_")
            r, c = int(base[0][1:]), int(base[1][1:])
            if (r, c) == exclude:
                continue
        except Exception:
            continue

        amp_mean = _parse_amps_from_csv(os.path.join(data_folder, fname))
        if amp_mean is None:
            continue
        all_means.append(float(np.mean(amp_mean)))
        all_vars.append(float(np.var(amp_mean)))

    if len(all_means) < 2:
        return None

    return {
        "n_points":       len(all_means),
        "mean_amplitude": round(float(np.mean(all_means)), 3),
        "std_amplitude":  round(float(np.std(all_means)), 3),
        "mean_variance":  round(float(np.mean(all_vars)), 4),
        "std_variance":   round(float(np.std(all_vars)), 4),
        "min_amplitude":  round(float(np.min(all_means)), 3),
        "max_amplitude":  round(float(np.max(all_means)), 3),
    }


def build_anomaly_prompt(current: dict, baseline: dict, row: int, col: int) -> str:
    """
    Build a compact comparison prompt for phi3.5:latest.
    Kept intentionally short for faster inference
    and speed up inference on CPU.
    """
    # How many std-devs is current from baseline mean?
    std = baseline["std_amplitude"] + 1e-9
    z_amp = (current["mean_amplitude"] - baseline["mean_amplitude"]) / std

    std_var = baseline["std_variance"] + 1e-9
    z_var   = (current["variance"] - baseline["mean_variance"]) / std_var

    return f"""
New grid point: row={row}, col={col}

CURRENT POINT stats:
  Mean amplitude : {current['mean_amplitude']}  (baseline avg: {baseline['mean_amplitude']} ± {baseline['std_amplitude']})
  Amplitude z-score: {z_amp:+.2f} std devs from room average
  CSI variance   : {current['variance']}  (baseline avg: {baseline['mean_variance']} ± {baseline['std_variance']})
  Variance z-score: {z_var:+.2f} std devs from room average
  Avg RSSI       : {current['avg_rssi']} dBm
  Zero IQ ratio  : {current['zero_ratio']:.1%}
  Packets parsed : {current['n_packets']}

ROOM BASELINE ({baseline['n_points']} other points):
  Amplitude range: {baseline['min_amplitude']} – {baseline['max_amplitude']}

Is this grid point anomalous compared to the rest of the room?
""".strip()


# ── LangGraph node ───────────────────────────────────────
def node_ai_anomaly_detection(state: dict) -> dict:
    """
    After each validated grid point:
    1. Compute stats for current point
    2. Load baseline from all other collected points
    3. Ask phi3.5:latest if it's anomalous + why
    """
    row, col  = state["last_point"]
    packets   = state["last_packets"]
    anomaly_log = state.get("anomaly_log", [])

    print(f"  [AI Anomaly] Analyzing r={row} c={col}...", flush=True)

    # ── Compute current point stats ──
    amp_mean = _parse_amps_from_packets(packets)
    rssi_vals = _parse_rssi_from_packets(packets)

    if amp_mean is None:
        print("  [AI Anomaly] Could not parse packets — skipping.")
        anomaly_log.append({"point": (row, col), "skipped": True})
        return {**state, "anomaly_log": anomaly_log}

    current_stats = {
        "mean_amplitude": round(float(np.mean(amp_mean)), 3),
        "variance":       round(float(np.var(amp_mean)), 4),
        "avg_rssi":       round(float(np.mean(rssi_vals)), 1) if rssi_vals else "N/A",
        "zero_ratio":     float(np.mean(amp_mean == 0)),
        "n_packets":      len(packets),
    }

    # ── Load baseline ──
    baseline = _load_baseline_stats(state["data_folder"], exclude=(row, col))

    if baseline is None:
        print(f"  [AI Anomaly] Only {len(anomaly_log)} prior points — need 2+ for comparison. Skipping.")
        anomaly_log.append({"point": (row, col), "verdict": "SKIPPED", "reason": "Not enough baseline data yet."})
        return {**state, "anomaly_log": anomaly_log}

    # ── Ask LLM ──
    user_prompt = build_anomaly_prompt(current_stats, baseline, row, col)
    response    = ask_llm(ANOMALY_SYSTEM_PROMPT, user_prompt)

    # ── Parse response ──
    verdict     = "UNKNOWN"
    confidence  = "LOW"
    reason      = response

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip().upper()
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    is_anomaly = "ANOMALOUS" in verdict

    # ── Print result ──
    if is_anomaly:
        color = "\033[93m"  # yellow
        icon  = "⚠"
    else:
        color = "\033[92m"  # green
        icon  = "✓"
    reset = "\033[0m"

    print(f"  {color}{icon} {verdict}{reset} (confidence: {confidence})")
    print(f"    → {reason}")

    log_entry = {
        "point":      (row, col),
        "verdict":    verdict,
        "confidence": confidence,
        "reason":     reason,
        "is_anomaly": is_anomaly,
        "stats":      current_stats,
    }
    anomaly_log.append(log_entry)

    return {**state, "anomaly_log": anomaly_log}
