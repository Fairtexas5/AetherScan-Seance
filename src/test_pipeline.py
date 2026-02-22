"""
test_pipeline.py
─────────────────────────────────────────────────────────
Self-contained test for the CSI pipeline.

Hardware mocked (no ESP32 / NodeMCU / serial port needed):
  ✓ Synthetic CSI packets are generated in-memory
  ✓ All user input() prompts are auto-answered
  ✓ Exercises every pipeline stage end-to-end

Ollama tested LIVE (if running) or mocked (if offline):
  ✓ Probes http://127.0.0.1:11434 before the pipeline starts
  ✓ If reachable → real phi3.5:latest calls go through
  ✓ If offline   → mock responses used, test still passes

Run:
    cd /Users/aditsg/Downloads/csi_project
    python test_pipeline.py

    # Force mock-AI even when Ollama is running:
    MOCK_AI=1 python test_pipeline.py
─────────────────────────────────────────────────────────
"""

import os
import sys
import time
import random
import types
import builtins
import socket

import numpy as np

# ── Ensure project root is on sys.path ───────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ═════════════════════════════════════════════════════════
#  SECTION 1 — Probe Ollama connectivity
# ═════════════════════════════════════════════════════════

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
FORCE_MOCK_AI = os.environ.get("MOCK_AI", "0") == "1"


def _ollama_is_reachable() -> bool:
    """Return True if the Ollama server TCP port is open."""
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=3):
            return True
    except OSError:
        return False


def test_ollama_connectivity() -> bool:
    """
    Step 1 of test sequence: verify Ollama is up and the model responds.
    Returns True if the real LLM should be used for the pipeline test.
    """
    print()
    print("─" * 62)
    print("  [OLLAMA TEST] Checking connectivity …")

    if FORCE_MOCK_AI:
        print("  [OLLAMA TEST]  MOCK_AI=1 env var set — skipping real Ollama.")
        return False

    if not _ollama_is_reachable():
        print(f"  [OLLAMA TEST]  ✗ Cannot reach {OLLAMA_HOST}:{OLLAMA_PORT}")
        print("  [OLLAMA TEST]  Start Ollama with:  ollama serve")
        print("  [OLLAMA TEST]  Continuing with MOCKED AI responses.\n")
        return False

    print(f"  [OLLAMA TEST]  ✓ TCP connection to {OLLAMA_HOST}:{OLLAMA_PORT} OK")

    # Send a minimal prompt to confirm the model is loaded
    try:
        from pipeline.ai_client import ask_llm  # real implementation

        print("  [OLLAMA TEST]  Sending ping prompt to phi3.5:latest …")
        t0 = time.time()
        response = ask_llm(
            system_prompt="You are a test assistant. Reply with exactly one word.",
            user_prompt="Respond with the single word: PONG",
        )
        elapsed = time.time() - t0

        if response.startswith("[AI"):
            # ai_client returns "[AI offline …]" or "[AI error: …]" on failure
            print(f"  [OLLAMA TEST]  ✗ Model error: {response}")
            print("  [OLLAMA TEST]  Continuing with MOCKED AI responses.\n")
            return False

        print(f"  [OLLAMA TEST]  ✓ Model replied in {elapsed:.1f}s: {response!r}")
        print("  [OLLAMA TEST]  Real Ollama calls will be used for the pipeline.\n")
        return True

    except Exception as exc:
        print(f"  [OLLAMA TEST]  ✗ Exception: {exc}")
        print("  [OLLAMA TEST]  Continuing with MOCKED AI responses.\n")
        return False


# ═════════════════════════════════════════════════════════
#  SECTION 2 — Always mock: serial / ESP32 hardware
# ═════════════════════════════════════════════════════════

mock_serial_utils = types.ModuleType("pipeline.serial_utils")


def _mock_auto_detect_port():
    print("    [HW MOCK] Serial port detection skipped — using synthetic port.")
    return "/dev/tty.mock0"


def _mock_open_serial(port, baudrate=921600):
    class FakeSerial:
        def close(self):
            pass
    return FakeSerial()


def _mock_flush_stale_data(ser, flush_seconds=0.5):
    pass


def _mock_read_csi_line(ser):
    """Emit one realistic synthetic CSI_DATA line (~500 lines/s)."""
    time.sleep(0.002)
    n_sub = 64
    rssi  = random.randint(-75, -45)
    ts    = int(time.time() * 1000)
    iq = []
    for _ in range(n_sub):
        iq.append(int(random.gauss(20, 8)))   # I
        iq.append(int(random.gauss(20, 8)))   # Q
    return f"CSI_DATA,{ts},{rssi},{n_sub},0,0," + ",".join(map(str, iq))


mock_serial_utils.auto_detect_port = _mock_auto_detect_port
mock_serial_utils.open_serial       = _mock_open_serial
mock_serial_utils.flush_stale_data  = _mock_flush_stale_data
mock_serial_utils.read_csi_line     = _mock_read_csi_line
sys.modules["pipeline.serial_utils"] = mock_serial_utils


# ── Auto-answer all input() prompts ──────────────────────
_input_idx = [0]
_real_input = builtins.input


def _auto_input(prompt=""):
    _input_idx[0] += 1
    print(f"    [INPUT MOCK] Auto-answering prompt #{_input_idx[0]}: {prompt!r}")
    return ""


builtins.input = _auto_input


# ═════════════════════════════════════════════════════════
#  SECTION 3 — Conditionally mock: ai_client
#              (only when Ollama is offline / MOCK_AI=1)
# ═════════════════════════════════════════════════════════

_ai_call_count  = {"n": 0}
_real_ai_active = {"v": False}   # set after Ollama probe


def _mock_ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Synthetic LLM responses — used when Ollama is unavailable."""
    _ai_call_count["n"] += 1
    n = _ai_call_count["n"]
    print(f"    [AI MOCK] Call #{n} — synthetic response.", flush=True)

    if "Is this data usable" in user_prompt:
        return "VERDICT: PASS\nREASON: Sufficient packets with good RSSI and amplitude variance."

    if "anomalous" in user_prompt.lower() or "anomalous" in system_prompt.lower():
        if n % 7 == 0:
            return (
                "VERDICT: ANOMALOUS\n"
                "CONFIDENCE: MEDIUM\n"
                "REASON: Elevated variance likely caused by a metal cabinet nearby."
            )
        return (
            "VERDICT: NORMAL\n"
            "CONFIDENCE: HIGH\n"
            "REASON: Signal consistent with open floor area."
        )

    return (
        "High-variance points near walls indicate RF reflections.\n"
        "Low-variance areas in the centre suggest open space.\n"
        "Suggestions:\n"
        "  1. Increase packets per point to 300 for smoother estimates.\n"
        "  2. Repeat the scan with the room cleared of movable objects."
    )


def _install_ai_mock():
    """Inject the mock ai_client into sys.modules."""
    mock_ai = types.ModuleType("pipeline.ai_client")
    mock_ai.ask_llm = _mock_ask_llm
    mock_ai.get_llm = lambda: (_ for _ in ()).throw(
        RuntimeError("get_llm() blocked in mock mode.")
    )
    sys.modules["pipeline.ai_client"] = mock_ai
    # Also patch already-imported references in sub-modules (if any)
    for mod_name in list(sys.modules):
        mod = sys.modules[mod_name]
        if hasattr(mod, "ask_llm") and mod is not mock_ai:
            mod.ask_llm = _mock_ask_llm


def _wrap_real_ai_for_counting():
    """Wrap the real ask_llm to count calls, without replacing behaviour."""
    from pipeline import ai_client as _real_ac
    _original = _real_ac.ask_llm

    def _counted(system_prompt, user_prompt):
        _ai_call_count["n"] += 1
        n = _ai_call_count["n"]
        print(f"    [AI REAL] Call #{n} → phi3.5:latest …", flush=True)
        result = _original(system_prompt, user_prompt)
        print(f"    [AI REAL] Call #{n} done.", flush=True)
        return result

    _real_ac.ask_llm = _counted


# ═════════════════════════════════════════════════════════
#  SECTION 4 — Pipeline import + test config
# ═════════════════════════════════════════════════════════

# NOTE: graph.py is imported AFTER the serial mock but the ai_client
# mock may or may not be installed depending on the Ollama probe above.
# We import the graph here (serial mock already in sys.modules) and
# handle ai_client patching right after the probe.


TEST_CONFIG = {
    "grid_rows": 3,
    "grid_cols": 3,

    "room_width_m":  3.0,
    "room_height_m": 3.0,

    # 60 packets × 2 ms synthetic delay ≈ 0.12 s per point (fast test)
    "packets_per_point": 60,

    "live_preview": True,
    "auto_retry":   False,   # keep the test short

    "data_folder":   "test_csi_data",
    "output_folder": "test_output",

    "serial_port": None,

    # Runtime state
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

    # AI state
    "anomaly_log":       [],
    "ai_report_path":    None,
    "ai_interpretation": None,
}


# ═════════════════════════════════════════════════════════
#  SECTION 5 — Main test runner
# ═════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 62)
    print("  CSI Pipeline — Full Test (Synthetic HW + Ollama probe)")
    print("═" * 62)

    # ── Step 1: Ollama probe ─────────────────────────────
    use_real_ai = test_ollama_connectivity()
    _real_ai_active["v"] = use_real_ai

    # ── Step 2: Apply ai_client mock if needed ───────────
    if use_real_ai:
        # Real Ollama is up — count calls but don't replace behaviour
        _wrap_real_ai_for_counting()
    else:
        # Ollama offline or forced mock — inject synthetic responses
        _install_ai_mock()

    # ── Step 3: Import pipeline (after all mocks are set) ─
    from pipeline.graph import build_pipeline

    print("─" * 62)
    print("  [PIPELINE TEST] Starting pipeline …")
    print(f"  Grid      : {TEST_CONFIG['grid_rows']} × {TEST_CONFIG['grid_cols']} "
          f"= {TEST_CONFIG['grid_rows'] * TEST_CONFIG['grid_cols']} points")
    print(f"  Pkts/pt   : {TEST_CONFIG['packets_per_point']} (synthetic)")
    print(f"  AI mode   : {'REAL (Ollama)' if use_real_ai else 'MOCK (offline fallback)'}")
    print(f"  Data      → {TEST_CONFIG['data_folder']}/")
    print(f"  Output    → {TEST_CONFIG['output_folder']}/")
    print()

    t0 = time.time()
    pipeline = build_pipeline()
    final_state = pipeline.invoke(TEST_CONFIG)
    elapsed = time.time() - t0

    # ── Step 4: Summary ──────────────────────────────────
    anomaly_log = final_state.get("anomaly_log", [])
    n_anomalies = sum(1 for a in anomaly_log if a.get("is_anomaly"))

    print()
    print("═" * 62)
    print("  TEST RESULTS")
    print("─" * 62)
    print(f"  Status      : {'✓ PASSED' if final_state.get('pipeline_done') else '✗ FAILED'}")
    print(f"  Time        : {elapsed:.1f}s")
    print(f"  AI mode     : {'real Ollama' if use_real_ai else 'mock (Ollama offline)'}")
    print(f"  AI calls    : {_ai_call_count['n']}")
    print(f"  Heatmap     : {final_state.get('heatmap_path', 'N/A')}")
    print(f"  AI Report   : {final_state.get('ai_report_path', 'N/A')}")
    print(f"  Anomalies   : {n_anomalies}/{len(anomaly_log)} grid points flagged")

    # Ollama ping result line
    if not FORCE_MOCK_AI:
        ollama_status = "✓ reachable" if use_real_ai else "✗ offline (mock used)"
        print(f"  Ollama      : {ollama_status}")

    print("═" * 62)
    print()

    # ── Step 5: Assertions ───────────────────────────────
    failures = []

    if not final_state.get("pipeline_done"):
        failures.append("pipeline_done flag not set")
    if not final_state.get("heatmap_path"):
        failures.append("heatmap_path is empty")
    elif not os.path.exists(final_state["heatmap_path"]):
        failures.append(f"heatmap PNG missing on disk: {final_state['heatmap_path']}")
    if not final_state.get("ai_report_path"):
        failures.append("ai_report_path is empty")
    elif not os.path.exists(final_state["ai_report_path"]):
        failures.append(f"AI report file missing: {final_state['ai_report_path']}")
    if _ai_call_count["n"] == 0:
        failures.append("no AI calls were made at all")

    if failures:
        print("  ✗ ASSERTION FAILURES:")
        for f in failures:
            print(f"    • {f}")
        print()
        sys.exit(1)
    else:
        print("  ✓ All assertions passed!\n")


if __name__ == "__main__":
    main()
