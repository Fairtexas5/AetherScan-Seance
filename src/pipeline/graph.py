"""
graph.py  (AI-enhanced version)
─────────────────────────────────────────────────────────
LangGraph state machine for the CSI pipeline.
No internet / No API key — uses local Ollama for all AI.

Full pipeline flow:
    setup
      ↓
    collect_point          ← user places ESP32, auto-collects packets
      ↓
    ai_validate_point      ← LLM decides if data quality is good
      ↓ (fail → retry collect_point, up to 3×)
      ↓ (pass)
    ai_anomaly_detection   ← ML flags unusual points, LLM explains why
      ↓
    live_preview           ← partial heatmap PNG updated
      ↓
    advance                ← move to next grid position
      ↓ (loop until all points done)
    process_heatmap        ← generate final heatmap PNG
      ↓
    ai_interpret_heatmap   ← LLM writes full plain-English report
      ↓
    END
─────────────────────────────────────────────────────────
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Tuple

from pipeline.nodes import (
    node_setup,
    node_collect_point,
    node_live_preview,
    node_advance,
    node_process_heatmap,
)
from pipeline.ai_validator   import node_ai_validate_point
from pipeline.ai_anomaly     import node_ai_anomaly_detection
from pipeline.ai_interpreter import node_ai_interpret_heatmap


# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────
class CSIPipelineState(TypedDict):
    # ── User config ──
    serial_port:       Optional[str]
    data_folder:       str
    output_folder:     str
    grid_rows:         int
    grid_cols:         int
    room_width_m:      float
    room_height_m:     float
    packets_per_point: int
    live_preview:      bool
    auto_retry:        bool

    # ── Runtime state ──
    setup_done:        bool
    current_row:       int
    current_col:       int
    last_packets:      List[str]
    last_point:        Optional[Tuple[int, int]]
    last_valid:        bool
    retry_count:       int
    collection_done:   bool
    heatmap_path:      Optional[str]
    pipeline_done:     bool

    # ── AI state ──
    anomaly_log:       List[dict]
    ai_report_path:    Optional[str]
    ai_interpretation: Optional[str]


# ─────────────────────────────────────────────
# Conditional Edge Routing
# ─────────────────────────────────────────────
def after_ai_validate(state: CSIPipelineState) -> str:
    if not state.get("last_valid", True):
        return "collect_point"
    return "ai_anomaly_detection"


def after_advance(state: CSIPipelineState) -> str:
    if state.get("collection_done", False):
        return "process_heatmap"
    return "collect_point"


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(CSIPipelineState)

    graph.add_node("setup",                node_setup)
    graph.add_node("collect_point",        node_collect_point)
    graph.add_node("ai_validate_point",    node_ai_validate_point)
    graph.add_node("ai_anomaly_detection", node_ai_anomaly_detection)
    graph.add_node("live_preview",         node_live_preview)
    graph.add_node("advance",              node_advance)
    graph.add_node("process_heatmap",      node_process_heatmap)
    graph.add_node("ai_interpret_heatmap", node_ai_interpret_heatmap)

    graph.set_entry_point("setup")

    graph.add_edge("setup",                "collect_point")
    graph.add_edge("collect_point",        "ai_validate_point")
    graph.add_edge("ai_anomaly_detection", "live_preview")
    graph.add_edge("live_preview",         "advance")
    graph.add_edge("process_heatmap",      "ai_interpret_heatmap")
    graph.add_edge("ai_interpret_heatmap", END)

    graph.add_conditional_edges(
        "ai_validate_point",
        after_ai_validate,
        {
            "collect_point":        "collect_point",
            "ai_anomaly_detection": "ai_anomaly_detection",
        }
    )
    graph.add_conditional_edges(
        "advance",
        after_advance,
        {
            "collect_point":   "collect_point",
            "process_heatmap": "process_heatmap",
        }
    )

    return graph.compile()
