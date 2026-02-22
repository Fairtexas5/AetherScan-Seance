"""
ai_interpreter.py
─────────────────────────────────────────────────────────
Final pipeline node: LLM interprets the completed RF heatmap.
Tuned for phi3.5:latest on MacBook Pro — concise prompt, 512 token output.

Where it fits:
    [process_heatmap] → [ai_interpret_heatmap] → END

Output: printed to terminal + saved to output/ai_report.txt
─────────────────────────────────────────────────────────
"""

import os
import numpy as np
from pipeline.ai_client import ask_llm
from pipeline import heatmap as hm


# ── Concise system prompt — phi3.5:latest works well with concise context ──
INTERPRETER_SYSTEM_PROMPT = """
You are a WiFi CSI (Channel State Information) expert helping a student
interpret room mapping results from an ESP32 experiment.

Given RF grid statistics, write a brief interpretation covering:
1. What high-variance areas (bright in heatmap) likely mean physically
2. What low-variance areas (dark) likely mean
3. What anomalies suggest about room objects/walls
4. 2 practical suggestions to improve the next run

Keep response under 300 words. Use plain English, no jargon.
""".strip()


def build_grid_stats(data_folder: str, grid_rows: int, grid_cols: int) -> dict:
    """Compute summary statistics of the completed RF grid."""
    grid = hm.build_grid(data_folder, grid_rows, grid_cols)
    valid = grid[~np.isnan(grid)]

    if len(valid) == 0:
        return {}

    flat = grid.flatten()
    sorted_i = np.argsort(flat)

    top5 = [f"r{i//grid_cols}c{i%grid_cols}" for i in sorted_i[-5:][::-1]
            if not np.isnan(flat[i])]
    bot5 = [f"r{i//grid_cols}c{i%grid_cols}" for i in sorted_i[:5]
            if not np.isnan(flat[i])]

    return {
        "collected":    int(np.sum(~np.isnan(grid))),
        "total":        grid_rows * grid_cols,
        "mean_var":     round(float(np.mean(valid)), 4),
        "std_var":      round(float(np.std(valid)), 4),
        "min_var":      round(float(np.min(valid)), 4),
        "max_var":      round(float(np.max(valid)), 4),
        "dynamic_range": round(float(np.max(valid) / (np.min(valid) + 1e-9)), 1),
        "high_var_pts": top5,
        "low_var_pts":  bot5,
    }


def summarize_anomalies(anomaly_log: list) -> str:
    if not anomaly_log:
        return "No anomaly data."
    detected = [a for a in anomaly_log if a.get("is_anomaly")]
    if not detected:
        return f"No anomalies — all {len(anomaly_log)} points were normal."
    lines = [f"{len(detected)} anomalous point(s):"]
    for a in detected:
        r, c = a["point"]
        lines.append(f"  r={r} c={c} ({a.get('confidence','?')} confidence): {a.get('reason','?')}")
    return "\n".join(lines)


# ── LangGraph node ───────────────────────────────────────
def node_ai_interpret_heatmap(state: dict) -> dict:
    print("\n" + "="*60)
    print("  AI Heatmap Interpretation (phi3.5:latest via Ollama)")
    print("="*60)

    stats = build_grid_stats(
        state["data_folder"], state["grid_rows"], state["grid_cols"]
    )
    if not stats:
        print("  [AI Interpreter] No data found.")
        return state

    anomaly_summary = summarize_anomalies(state.get("anomaly_log", []))

    # Compact prompt for phi3.5:latest
    user_prompt = (
        f"Room: {state['room_width_m']*100:.0f}cm × {state['room_height_m']*100:.0f}cm | "
        f"Grid: {state['grid_rows']}×{state['grid_cols']} | "
        f"Points: {stats['collected']}/{stats['total']}\n\n"
        f"RF Grid Stats (CSI amplitude variance per position):\n"
        f"  Mean={stats['mean_var']}, Std={stats['std_var']}, "
        f"  Min={stats['min_var']}, Max={stats['max_var']}, "
        f"  Dynamic range={stats['dynamic_range']}×\n"
        f"  Highest variance points (near walls/objects): {', '.join(stats['high_var_pts'])}\n"
        f"  Lowest variance points (open space): {', '.join(stats['low_var_pts'])}\n\n"
        f"Anomaly summary:\n{anomaly_summary}\n\n"
        f"Please interpret these results."
    )

    interpretation = ask_llm(INTERPRETER_SYSTEM_PROMPT, user_prompt)

    # Print
    print()
    print("  ┌─ AI Report ────────────────────────────────────────")
    for line in interpretation.splitlines():
        print(f"  │ {line}")
    print("  └────────────────────────────────────────────────────\n")

    # Save
    report_path = os.path.join(state["output_folder"], "ai_report.txt")
    os.makedirs(state["output_folder"], exist_ok=True)
    with open(report_path, "w") as f:
        f.write("CSI Room Mapping — AI Interpretation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Room: {state['room_width_m']*100:.0f}cm × {state['room_height_m']*100:.0f}cm\n")
        f.write(f"Grid: {state['grid_rows']}×{state['grid_cols']} "
                f"({stats['collected']}/{stats['total']} points)\n\n")
        f.write("Grid Statistics:\n")
        for k, v in stats.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nAnomaly Summary:\n{anomaly_summary}\n\n")
        f.write("AI Interpretation:\n" + "-"*40 + "\n")
        f.write(interpretation + "\n")

    print(f"  ✓ AI report saved → {report_path}")
    return {**state, "ai_report_path": report_path, "ai_interpretation": interpretation}
