"""
csi_visualizer.py
─────────────────────────────────────────────────────────
Plotly-based CSI visualization suite.

Generates three interactive HTML charts from collected CSI data:

  1. 2D Heatmap     —  px.imshow  — room grid coloured by RF intensity
  2. 3D Surface     —  go.Surface — same grid extruded as a 3D surface
  3. 3D Room        —  go.Scatter3d + box — measurement points in 3D space

All charts are saved as standalone HTML (no server needed).

Standalone usage:
    python src/pipeline/csi_visualizer.py
    python src/pipeline/csi_visualizer.py --data csi_data --output output --room 5x4x2.5

Programmatic usage:
    from pipeline.csi_visualizer import build_all_charts
    build_all_charts(grid2d, room_w=5, room_d=4, room_h=2.5, output_folder="output")

─────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "plotly is required.  Install with:  pip install plotly"
    )

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
#  Chart 1 — 2D Interactive Heatmap
# ═══════════════════════════════════════════════════════════════════════

def chart_2d_heatmap(
    grid2d: np.ndarray,
    room_w: float,
    room_d: float,
    output_folder: str,
    title: str = "CSI 2D RF Heatmap",
) -> str:
    """
    Interactive 2D heatmap using px.imshow.
    Axes are in real room metres. NaN cells shown in grey.
    Returns path to saved HTML.
    """
    rows, cols = grid2d.shape

    # Build axis tick labels in metres
    x_labels = [f"{room_w * c / max(cols - 1, 1) * 100:.0f} cm" for c in range(cols)]
    y_labels = [f"{room_d * r / max(rows - 1, 1) * 100:.0f} cm" for r in range(rows)]

    # Normalise (keep NaN as-is; imshow handles them)
    g = grid2d.copy().astype(float)
    vmin = float(np.nanmin(g)) if not np.all(np.isnan(g)) else 0.0
    vmax = float(np.nanmax(g)) if not np.all(np.isnan(g)) else 1.0

    fig = px.imshow(
        g,
        labels={"x": "Room Width (cm)", "y": "Room Depth (cm)", "color": "RF Intensity"},
        x=x_labels,
        y=y_labels,
        color_continuous_scale="Jet",
        zmin=vmin,
        zmax=vmax,
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        coloraxis_colorbar=dict(
            title="CSI Variance<br>(RF Intensity)",
            tickformat=".3f",
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor="#111111",
        paper_bgcolor="#1a1a2e",
        font=dict(color="white", family="Inter, Arial, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed")

    out_path = os.path.join(output_folder, "csi_2d_heatmap.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[visualizer] 2D heatmap  → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
#  Chart 2 — 3D Surface Heatmap
# ═══════════════════════════════════════════════════════════════════════

def chart_3d_surface(
    grid2d: np.ndarray,
    room_w: float,
    room_d: float,
    output_folder: str,
    title: str = "CSI 3D Surface Heatmap",
) -> str:
    """
    3D surface using go.Surface.
    X = room width, Y = room depth, Z = normalized RF intensity.
    Returns path to saved HTML.
    """
    rows, cols = grid2d.shape

    x = np.linspace(0, room_w, cols)   # metres across width
    y = np.linspace(0, room_d, rows)   # metres across depth
    X, Y = np.meshgrid(x, y)

    g = grid2d.copy().astype(float)

    # Fill NaN with minimum for surface coherence
    nan_mask = np.isnan(g)
    g_min    = float(np.nanmin(g)) if not np.all(nan_mask) else 0.0
    g[nan_mask] = g_min

    # Normalize to [0, 1]
    g_max = float(np.nanmax(g))
    if g_max > g_min:
        g_norm = (g - g_min) / (g_max - g_min)
    else:
        g_norm = np.zeros_like(g)

    # Grey out NaN cells on the surface
    custom_color = g_norm.copy()
    custom_color[nan_mask] = -0.05   # slightly below 0 → shows as grey

    colorscale = [
        [0.00, "#444444"],  # uncollected (NaN shown as -0.05)
        [0.05, "#06003c"],
        [0.25, "#1a0099"],
        [0.50, "#0066ff"],
        [0.75, "#00cccc"],
        [0.90, "#ffee00"],
        [1.00, "#ff2200"],
    ]

    surf = go.Surface(
        x=X, y=Y, z=g_norm,
        surfacecolor=custom_color,
        colorscale=colorscale,
        cmin=0, cmax=1,
        colorbar=dict(
            title=dict(text="RF Intensity<br>(normalized)", side="right"),
            tickformat=".2f",
            len=0.7,
        ),
        hovertemplate=(
            "Width: %{x}<br>"
            "Depth: %{y}<br>"
            "RF:    %{z:.4f}<extra></extra>"
        ),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))
        ),
    )

    fig = go.Figure(data=[surf])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color="white")),
        scene=dict(
            xaxis=dict(title="Width (cm)", backgroundcolor="#111111",
                       gridcolor="#333", showbackground=True),
            yaxis=dict(title="Depth (cm)", backgroundcolor="#111111",
                       gridcolor="#333", showbackground=True),
            zaxis=dict(title="RF Intensity", backgroundcolor="#111111",
                       gridcolor="#333", showbackground=True, range=[-0.1, 1.1]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            bgcolor="#111111",
        ),
        paper_bgcolor="#1a1a2e",
        font=dict(color="white", family="Inter, Arial, sans-serif"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out_path = os.path.join(output_folder, "csi_3d_surface.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[visualizer] 3D surface  → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
#  Chart 3 — 3D Room Structure Plot
# ═══════════════════════════════════════════════════════════════════════

def _room_wireframe(room_w: float, room_d: float, room_h: float) -> go.Scatter3d:
    """Build 8-corner room wireframe as a single Scatter3d trace."""
    corners = [
        (0,      0,      0),
        (room_w, 0,      0),
        (room_w, room_d, 0),
        (0,      room_d, 0),
        (0,      0,      0),       # close bottom
        (0,      0,      room_h),
        (room_w, 0,      room_h),
        (room_w, room_d, room_h),
        (0,      room_d, room_h),
        (0,      0,      room_h),  # close top
    ]
    # Add vertical pillars
    pillars = [
        (room_w, 0,      0), (room_w, 0,      room_h), (None, None, None),
        (room_w, room_d, 0), (room_w, room_d, room_h), (None, None, None),
        (0,      room_d, 0), (0,      room_d, room_h), (None, None, None),
    ]
    pts = list(corners) + pillars
    xs, ys, zs = zip(*[(p[0], p[1], p[2]) if p[0] is not None else (None, None, None)
                       for p in pts])
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="#4488ff", width=3),
        name="Room boundary",
        hoverinfo="skip",
    )


def chart_3d_room_structure(
    grid2d: np.ndarray,
    room_w: float,
    room_d: float,
    room_h: float,
    output_folder: str,
    title: str = "CSI 3D Room Structure",
) -> str:
    """
    3D room structure — wireframe room box with measurement points as spheres
    coloured by signal strength (RF intensity). Uncollected cells shown in grey.
    Returns path to saved HTML.
    """
    rows, cols = grid2d.shape

    # Build measurement point coordinates
    pt_x, pt_y, pt_z, pt_val, pt_text = [], [], [], [], []

    for r in range(rows):
        for c in range(cols):
            wx  = room_w * c / max(cols - 1, 1)
            wy  = room_d * r / max(rows - 1, 1)
            wz  = room_h * 0.05          # measurement height (5 cm above floor)
            val = float(grid2d[r, c]) if not np.isnan(grid2d[r, c]) else None
            pt_x.append(wx)
            pt_y.append(wy)
            pt_z.append(wz)
            pt_val.append(val)
            label = f"Row {r}, Col {c}<br>RF: {val:.4f}" if val is not None else f"Row {r}, Col {c}<br>Not collected"
            pt_text.append(label)

    # Separate collected vs uncollected
    c_x   = [pt_x[i]   for i, v in enumerate(pt_val) if v is not None]
    c_y   = [pt_y[i]   for i, v in enumerate(pt_val) if v is not None]
    c_z   = [pt_z[i]   for i, v in enumerate(pt_val) if v is not None]
    c_val = [v          for v in pt_val if v is not None]
    c_txt = [pt_text[i] for i, v in enumerate(pt_val) if v is not None]

    u_x   = [pt_x[i]   for i, v in enumerate(pt_val) if v is None]
    u_y   = [pt_y[i]   for i, v in enumerate(pt_val) if v is None]
    u_z   = [pt_z[i]   for i, v in enumerate(pt_val) if v is None]
    u_txt = [pt_text[i] for i, v in enumerate(pt_val) if v is None]

    # Normalize collected values for colour
    if c_val:
        vmin = min(c_val)
        vmax = max(c_val)
        c_norm = [(v - vmin) / max(vmax - vmin, 1e-9) for v in c_val]
    else:
        c_norm = []

    traces = [_room_wireframe(room_w, room_d, room_h)]

    # Collected points (coloured spheres)
    if c_x:
        traces.append(go.Scatter3d(
            x=c_x, y=c_y, z=c_z,
            mode="markers",
            marker=dict(
                size=10,
                color=c_norm,
                colorscale=[
                    [0.0,  "#06003c"],
                    [0.25, "#0066ff"],
                    [0.5,  "#00cccc"],
                    [0.75, "#ffee00"],
                    [1.0,  "#ff2200"],
                ],
                cmin=0, cmax=1,
                opacity=0.9,
                colorbar=dict(
                    title=dict(text="RF Intensity", side="right"),
                    tickformat=".2f",
                    x=1.02, len=0.6,
                ),
                symbol="circle",
            ),
            text=c_txt,
            hovertemplate="%{text}<extra></extra>",
            name="Collected points",
        ))

    # Uncollected points (grey)
    if u_x:
        traces.append(go.Scatter3d(
            x=u_x, y=u_y, z=u_z,
            mode="markers",
            marker=dict(size=8, color="#555555", opacity=0.5, symbol="circle"),
            text=u_txt,
            hovertemplate="%{text}<extra></extra>",
            name="Not collected",
        ))

    # RF intensity column bars (thin vertical lines from floor to point height)
    if c_x and c_val:
        vmax_val = max(c_val) if c_val else 1.0
        for i, (cx, cy, cv) in enumerate(zip(c_x, c_y, c_val)):
            bar_h = (cv - (min(c_val) if c_val else 0)) / max(vmax_val - (min(c_val) if c_val else 0), 1e-9)
            bar_h = bar_h * room_h * 0.4   # max 40% of room height
            traces.append(go.Scatter3d(
                x=[cx, cx], y=[cy, cy], z=[0, bar_h],
                mode="lines",
                line=dict(color=f"rgba(100,180,255,0.4)", width=4),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color="white")),
        scene=dict(
            xaxis=dict(title=f"Width (0–{room_w*100:.0f}cm)", range=[0, room_w],
                       backgroundcolor="#111111", gridcolor="#333", showbackground=True),
            yaxis=dict(title=f"Depth (0–{room_d*100:.0f}cm)", range=[0, room_d],
                       backgroundcolor="#111111", gridcolor="#333", showbackground=True),
            zaxis=dict(title=f"Height (0–{room_h*100:.0f}cm)", range=[0, room_h],
                       backgroundcolor="#111111", gridcolor="#333", showbackground=True),
            aspectmode="manual",
            aspectratio=dict(x=room_w / room_h, y=room_d / room_h, z=1.0),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            bgcolor="#111111",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white"),
            x=0.01, y=0.99,
        ),
        paper_bgcolor="#1a1a2e",
        font=dict(color="white", family="Inter, Arial, sans-serif"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out_path = os.path.join(output_folder, "csi_3d_room.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[visualizer] 3D room     → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
#  Master entry — build all three charts
# ═══════════════════════════════════════════════════════════════════════

def build_all_charts(
    grid2d: np.ndarray,
    room_w: float,
    room_d: float,
    room_h: float,
    output_folder: str,
    label: str = "",
) -> dict:
    """
    Generate all three Plotly charts from a 2D CSI grid.
    Returns dict of {chart_name: html_path}.
    """
    os.makedirs(output_folder, exist_ok=True)
    suffix = f" — {label}" if label else ""

    paths = {}
    paths["2d_heatmap"]   = chart_2d_heatmap(
        grid2d, room_w, room_d, output_folder,
        title=f"CSI 2D RF Heatmap{suffix}"
    )
    paths["3d_surface"]   = chart_3d_surface(
        grid2d, room_w, room_d, output_folder,
        title=f"CSI 3D Surface Heatmap{suffix}"
    )
    paths["3d_room"]      = chart_3d_room_structure(
        grid2d, room_w, room_d, room_h, output_folder,
        title=f"CSI 3D Room Structure{suffix}"
    )
    return paths


# ═══════════════════════════════════════════════════════════════════════
#  Standalone CLI — load csi_data/ folder and render
# ═══════════════════════════════════════════════════════════════════════

def _parse_room(s: str) -> tuple:
    parts = s.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected WxDxH in cm (e.g. 500x400x250)")
    return tuple(float(p) / 100.0 for p in parts)   # cm → m internally


def main():
    parser = argparse.ArgumentParser(
        prog="csi_visualizer.py",
        description="Generate Plotly CSI charts from collected csi_data/ folder",
    )
    parser.add_argument("--data",   default="csi_data",
                        help="Folder containing r??_c??.csv files (default: csi_data)")
    parser.add_argument("--output", default="output",
                        help="Output folder for HTML files (default: output)")
    parser.add_argument("--room",   default="500x400x250",
                        help="Room dimensions WxDxH in centimetres (default: 500x400x250)")
    parser.add_argument("--rows",   type=int, default=None,
                        help="Grid rows (auto-detected from CSV files if omitted)")
    parser.add_argument("--cols",   type=int, default=None,
                        help="Grid cols (auto-detected from CSV files if omitted)")
    args = parser.parse_args()

    room_w, room_d, room_h = _parse_room(args.room)

    # Load grid via heatmap module
    from pipeline.heatmap import build_grid

    if args.rows and args.cols:
        grid_rows, grid_cols = args.rows, args.cols
    else:
        # Auto-detect from CSV filenames in data folder
        import glob, re
        files = glob.glob(os.path.join(args.data, "r??_c??.csv"))
        if not files:
            print(f"[visualizer] No CSV files found in '{args.data}'")
            print("  Run the AI pipeline first to collect data.")
            sys.exit(1)
        max_r = max_c = 0
        for f in files:
            m = re.search(r"r(\d+)_c(\d+)\.csv", os.path.basename(f))
            if m:
                max_r = max(max_r, int(m.group(1)))
                max_c = max(max_c, int(m.group(2)))
        grid_rows = max_r + 1
        grid_cols = max_c + 1
        print(f"[visualizer] Auto-detected grid: {grid_rows} rows × {grid_cols} cols")

    print(f"[visualizer] Loading CSI data from: {args.data}/")
    grid2d = build_grid(args.data, grid_rows, grid_cols)

    print()
    print(f"[visualizer] Room  : {room_w}m × {room_d}m × {room_h}m")
    print(f"[visualizer] Grid  : {grid_rows} rows × {grid_cols} cols")
    filled = int(np.sum(~np.isnan(grid2d)))
    print(f"[visualizer] Cells : {filled}/{grid_rows * grid_cols} collected")
    print()

    paths = build_all_charts(grid2d, room_w, room_d, room_h, args.output)

    print()
    print("=" * 60)
    print("  Charts ready — open in any browser:")
    for name, path in paths.items():
        print(f"    [{name:12s}]  {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
