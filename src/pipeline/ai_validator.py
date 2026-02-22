"""
ai_validator.py
─────────────────────────────────────────────────────────
AI-powered CSI data quality validator using phi3.5:latest via Ollama.
Tuned for MacBook Pro — prompts kept concise for fast inference.

Where it fits:
    [collect_point] → [ai_validate_point] → [ai_anomaly_detection]
─────────────────────────────────────────────────────────
"""

import numpy as np
from typing import List, Tuple
from pipeline.ai_client import ask_llm
from pipeline.csi_parser import parse_line


# ── Short prompt = faster on CPU ──────────────────────────
VALIDATOR_SYSTEM_PROMPT = """
You are a WiFi CSI (Channel State Information) signal expert.
Assess if CSI data collected at a grid position is usable for room mapping.

PASS if: enough packets (50+), RSSI above -90 dBm, IQ values not all zero, some variance exists.
FAIL if: too few packets, no signal, all-zero IQ values, or obvious hardware disconnection.

Respond in this EXACT format (2 lines only):
VERDICT: PASS   or   VERDICT: FAIL
REASON: <one sentence>
""".strip()


def compute_stats(packets: List[str]) -> dict:
    """Extract key signal statistics from raw CSI packet lines."""
    rssi_vals, all_amps = [], []
    parsed_count = 0
    n_sub = None

    for line in packets:
        result = parse_line(line)
        if result is None:
            continue
        rssi_vals.append(result["rssi"])
        n_sub = result["n_sub"]
        all_amps.extend(result["amplitude"].tolist())
        parsed_count += 1

    stats = {
        "total":  len(packets),
        "parsed": parsed_count,
        "n_sub":  n_sub or "unknown",
    }
    if rssi_vals:
        stats["avg_rssi"] = round(float(np.mean(rssi_vals)), 1)
        stats["min_rssi"] = int(min(rssi_vals))
    if all_amps:
        arr = np.array(all_amps)
        stats["avg_amp"]    = round(float(np.mean(arr)), 3)
        stats["variance"]   = round(float(np.var(arr)), 4)
        stats["zero_ratio"] = round(float(np.mean(arr == 0)), 3)

    return stats


def ai_validate(packets: List[str], row: int, col: int) -> Tuple[bool, str]:
    """Ask local LLM to validate data quality. Returns (is_valid, reason)."""
    stats = compute_stats(packets)

    # Build compact prompt
    user_prompt = (
        f"Grid position: row={row}, col={col}\n"
        f"Packets total: {stats['total']} | Parsed: {stats['parsed']}\n"
        f"Subcarriers: {stats.get('n_sub', '?')} | "
        f"Avg RSSI: {stats.get('avg_rssi', 'N/A')} dBm | "
        f"Min RSSI: {stats.get('min_rssi', 'N/A')} dBm\n"
        f"Avg amplitude: {stats.get('avg_amp', 'N/A')} | "
        f"Variance: {stats.get('variance', 'N/A')} | "
        f"Zero ratio: {stats.get('zero_ratio', 'N/A')}\n\n"
        f"Is this data usable?"
    )

    response = ask_llm(VALIDATOR_SYSTEM_PROMPT, user_prompt)

    # Parse
    is_valid = True
    reason   = "OK"
    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict  = line.split(":", 1)[1].strip().upper()
            is_valid = "PASS" in verdict
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return is_valid, reason


# ── LangGraph node ───────────────────────────────────────
def node_ai_validate_point(state: dict) -> dict:
    row, col = state["last_point"]
    packets  = state["last_packets"]

    print(f"  [AI Validator] Checking data quality for r={row} c={col}...")
    is_valid, reason = ai_validate(packets, row, col)

    color  = "\033[92m" if is_valid else "\033[91m"
    status = "✓ PASS" if is_valid else "✗ FAIL"
    reset  = "\033[0m"
    print(f"  [{color}{status}{reset}] {reason}")

    if not is_valid and state.get("auto_retry", True):
        retry_count = state.get("retry_count", 0) + 1
        if retry_count <= 3:
            print(f"  ↻ Retry {retry_count}/3 — please reposition ESP32 at r={row} c={col}")
            return {**state, "last_valid": False, "retry_count": retry_count}
        else:
            print(f"  ⚠ Max retries reached. Accepting data as-is.")

    return {**state, "last_valid": True, "retry_count": 0}
