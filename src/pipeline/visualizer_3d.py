"""
visualizer_3d.py
─────────────────────────────────────────────────────────
Interactive Plotly 3D room heatmap for the WiFi CSI Mapper.

Two view modes:
  1. launch_3d_viewer(voxel_grid, …)
       Render the full 3D scatter + animated XY/XZ/YZ slices.
       Opens in the default browser.  Also saves room_3d.html.

  2. launch_from_grid2d(grid, …)
       Takes the existing 2D heatmap numpy grid (rows × cols)
       and extrudes it into a simple 3D surface for a quick
       preview after the LangGraph pipeline finishes.

Color scale: 'plasma'  (dark purple → bright yellow)
  dark  = low RF intensity   → open space, clean signal path
  bright = high RF intensity → walls, corners, furniture

Usage:
    from pipeline.visualizer_3d import launch_3d_viewer
    launch_3d_viewer(voxel_grid, room_w=5, room_d=4, room_h=2.5,
                     output_folder="output")
─────────────────────────────────────────────────────────
"""

import os
import numpy as np

# ── Plotly (required) ───────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


def _check_plotly():
    if not _PLOTLY_OK:
        raise ImportError(
            "[visualizer_3d] plotly is not installed.\n"
            "  Fix: pip install plotly"
        )


# ──────────────────────────────────────────────────────────────────────
# PUBLIC: Full 3D viewer (voxel scatter + animated slices)
# ──────────────────────────────────────────────────────────────────────
def launch_3d_viewer(
    voxel_grid: np.ndarray,
    room_w: float = 5.0,
    room_d: float = 4.0,
    room_h: float = 2.5,
    output_folder: str = "output",
    title: str = "WiFi CSI 3D Room Map",
):
    """
    Open an interactive Plotly figure with:
      • Tab 1 — 3D volume scatter: every voxel plotted as a
                coloured, semi-transparent marker scaled by intensity.
      • Tab 2 — Animated slice view: XY / XZ / YZ cross-sections
                with a Play button that sweeps through each axis.

    Parameters
    ----------
    voxel_grid   : np.ndarray shape (nW, nD, nH), values in [0, 1]
    room_w/d/h   : physical dimensions in metres (for axis labels)
    output_folder: directory to save room_3d.html
    title        : figure title string
    """
    _check_plotly()
    os.makedirs(output_folder, exist_ok=True)

    nW, nD, nH = voxel_grid.shape

    # ── Map voxel indices → real-world metres ───────────────────────
    xs = np.linspace(0, room_w, nW)
    ys = np.linspace(0, room_d, nD)
    zs = np.linspace(0, room_h, nH)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    vals = voxel_grid  # already [0,1]

    # Flatten for scatter
    x_flat = XX.ravel()
    y_flat = YY.ravel()
    z_flat = ZZ.ravel()
    v_flat = vals.ravel()

    # Keep only voxels above a threshold to avoid clutter
    thresh = 0.05
    mask = v_flat >= thresh
    x_flat, y_flat, z_flat, v_flat = (
        x_flat[mask], y_flat[mask], z_flat[mask], v_flat[mask]
    )

    # Marker size scaled by intensity
    marker_size = 2 + v_flat * 6

    # ── Figure 1: 3D Volume Scatter ─────────────────────────────────
    fig_3d = go.Figure(
        data=[
            go.Scatter3d(
                x=x_flat, y=y_flat, z=z_flat,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=v_flat,
                    colorscale="plasma",
                    opacity=0.55,
                    colorbar=dict(
                        title="RF Intensity",
                        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                        ticktext=["Low", "", "Mid", "", "High"],
                    ),
                    cmin=0, cmax=1,
                ),
                text=[f"({x*100:.0f}cm, {y*100:.0f}cm, {z*100:.0f}cm)<br>intensity={v:.2f}"
                      for x, y, z, v in zip(x_flat, y_flat, z_flat, v_flat)],
                hoverinfo="text",
                name="RF Voxel",
            )
        ]
    )
    fig_3d.update_layout(
        title=dict(text=f"{title} — 3D Volume", font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="Width (cm)", range=[0, room_w]),
            yaxis=dict(title="Depth (cm)", range=[0, room_d]),
            zaxis=dict(title="Height (cm)", range=[0, room_h]),
            bgcolor="#0d0d1a",
            xaxis_backgroundcolor="#0d0d1a",
            yaxis_backgroundcolor="#0d0d1a",
            zaxis_backgroundcolor="#0d0d1a",
        ),
        paper_bgcolor="#111122",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # ── Figure 2: Animated Slice Viewer ─────────────────────────────
    fig_slices = _build_animated_slices(voxel_grid, xs, ys, zs, room_w, room_d, room_h, title)

    # ── Combined HTML ────────────────────────────────────────────────
    html_path = os.path.join(output_folder, "room_3d.html")
    html3d     = fig_3d.to_html(full_html=False, include_plotlyjs="cdn")
    html_slices = fig_slices.to_html(full_html=False, include_plotlyjs=False)

    combined_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>{title}</title>
  <style>
    body  {{ margin:0; background:#0a0a18; color:#e0e0ff;
             font-family: 'Segoe UI', Arial, sans-serif; }}
    h1    {{ text-align:center; padding:20px 0 5px;
             font-size:1.6em; color:#c8b4ff; letter-spacing:1px; }}
    p.sub {{ text-align:center; color:#888; margin:0 0 8px; font-size:.9em; }}
    .tab-bar {{ display:flex; justify-content:center; gap:12px; padding:8px; }}
    .tab-btn {{ padding:8px 22px; border:2px solid #6644aa;
                border-radius:20px; background:#1a1a33; color:#c8b4ff;
                cursor:pointer; font-size:.95em; transition:.2s; }}
    .tab-btn:hover, .tab-btn.active {{ background:#6644aa; color:#fff; }}
    .tab-pane {{ display:none; }}
    .tab-pane.active {{ display:block; }}
    .plot-box {{ width:98%; margin:0 auto; }}
  </style>
</head>
<body>
  <h1>📡 {title}</h1>
  <p class='sub'>WiFi CSI Radio Frequency Mapping — Interactive 3D Viewer</p>
  <div class='tab-bar'>
    <button class='tab-btn active' onclick="showTab('vol')">🧊 3D Volume</button>
    <button class='tab-btn'        onclick="showTab('slc')">🔪 Animated Slices</button>
  </div>
  <div id='vol' class='tab-pane active plot-box'>{html3d}</div>
  <div id='slc' class='tab-pane plot-box'>{html_slices}</div>
  <script>
    function showTab(id) {{
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn' ).forEach(b => b.classList.remove('active'));
      document.getElementById(id).classList.add('active');
      event.target.classList.add('active');
    }}
  </script>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(combined_html)
    print(f"[visualizer_3d] 3D viewer saved → {html_path}")

    # Open in browser
    import webbrowser, pathlib
    webbrowser.open(pathlib.Path(os.path.abspath(html_path)).as_uri())
    return html_path


# ──────────────────────────────────────────────────────────────────────
# INTERNAL: Build animated slice figure
# ──────────────────────────────────────────────────────────────────────
def _build_animated_slices(voxel_grid, xs, ys, zs, room_w, room_d, room_h, title):
    """Build a Plotly figure with 3 rows (XY/XZ/YZ) and animated frames."""
    nW, nD, nH = voxel_grid.shape

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Floor Plan (XY) — Height Sweep",
                        "Front Section (XZ) — Depth Sweep",
                        "Side Section (YZ) — Width Sweep"],
        horizontal_spacing=0.06,
    )

    # Initial slice at midpoint
    mid_h = nH // 2
    mid_d = nD // 2
    mid_w = nW // 2

    def _heatmap(z_data, x_range, y_range, row, col):
        return go.Heatmap(
            z=z_data,
            x=x_range, y=y_range,
            colorscale="plasma",
            zmin=0, zmax=1,
            showscale=(col == 3),
            colorbar=dict(title="RF", x=1.01) if col == 3 else None,
        )

    # Initial traces
    fig.add_trace(_heatmap(voxel_grid[:, :, mid_h].T, xs, ys, 1, 1), row=1, col=1)
    fig.add_trace(_heatmap(voxel_grid[:, mid_d, :].T, xs, zs, 1, 2), row=1, col=2)
    fig.add_trace(_heatmap(voxel_grid[mid_w, :, :].T, ys, zs, 1, 3), row=1, col=3)

    # Build frames — sweep height, then depth, then width
    frames = []
    slider_steps = []

    # Height sweep (floor → ceiling)
    for iz in range(nH):
        label = f"H={zs[iz]*100:.0f}cm"
        f = go.Frame(
            data=[
                _heatmap(voxel_grid[:, :, iz].T,    xs, ys, 1, 1),
                _heatmap(voxel_grid[:, mid_d, :].T, xs, zs, 1, 2),
                _heatmap(voxel_grid[mid_w, :, :].T, ys, zs, 1, 3),
            ],
            name=label,
            traces=[0, 1, 2],
        )
        frames.append(f)
        slider_steps.append(dict(
            args=[[label], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
            label=label, method="animate"
        ))

    # Depth sweep (front → back)
    for iy in range(nD):
        label = f"D={ys[iy]*100:.0f}cm"
        f = go.Frame(
            data=[
                _heatmap(voxel_grid[:, :, mid_h].T, xs, ys, 1, 1),
                _heatmap(voxel_grid[:, iy, :].T,    xs, zs, 1, 2),
                _heatmap(voxel_grid[mid_w, :, :].T, ys, zs, 1, 3),
            ],
            name=label,
            traces=[0, 1, 2],
        )
        frames.append(f)
        slider_steps.append(dict(
            args=[[label], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
            label=label, method="animate"
        ))

    # Width sweep (left → right)
    for ix in range(nW):
        label = f"W={xs[ix]*100:.0f}cm"
        f = go.Frame(
            data=[
                _heatmap(voxel_grid[:, :, mid_h].T, xs, ys, 1, 1),
                _heatmap(voxel_grid[:, mid_d, :].T, xs, zs, 1, 2),
                _heatmap(voxel_grid[ix, :, :].T,    ys, zs, 1, 3),
            ],
            name=label,
            traces=[0, 1, 2],
        )
        frames.append(f)
        slider_steps.append(dict(
            args=[[label], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
            label=label, method="animate"
        ))

    fig.frames = frames

    fig.update_layout(
        title=dict(text=f"{title} — Animated Slices", font=dict(size=15, color="white")),
        paper_bgcolor="#111122",
        plot_bgcolor="#0d0d1a",
        font=dict(color="white"),
        height=480,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.08, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 80, "redraw": True},
                                  "fromcurrent": True, "mode": "immediate"}]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ]
        )],
        sliders=[dict(
            steps=slider_steps,
            currentvalue=dict(prefix="Slice: ", font=dict(color="white")),
            pad=dict(t=50),
        )],
    )

    # Axis labels
    fig.update_xaxes(title_text="Width (cm)", row=1, col=1)
    fig.update_yaxes(title_text="Depth (cm)", row=1, col=1)
    fig.update_xaxes(title_text="Width (cm)", row=1, col=2)
    fig.update_yaxes(title_text="Height (cm)", row=1, col=2)
    fig.update_xaxes(title_text="Depth (cm)", row=1, col=3)
    fig.update_yaxes(title_text="Height (cm)", row=1, col=3)

    return fig


# ──────────────────────────────────────────────────────────────────────
# PUBLIC: Quick 3D from 2D grid (used after LangGraph heatmap pipeline)
# ──────────────────────────────────────────────────────────────────────
def launch_from_grid2d(
    grid2d: np.ndarray,
    room_w: float = 5.0,
    room_d: float = 4.0,
    room_h: float = 2.5,
    output_folder: str = "output",
    title: str = "WiFi CSI 3D Room Map",
):
    """
    Take the existing 2D CSI variance grid (nRows × nCols) and
    extrude it into a simple 3D surface + scatter for a quick Plotly viewer.
    Used at the end of the LangGraph pipeline when real ESP32 data was collected.
    """
    _check_plotly()

    rows, cols = grid2d.shape
    # Normalise
    gmin, gmax = np.nanmin(grid2d), np.nanmax(grid2d)
    g = np.where(np.isnan(grid2d), 0.0, (grid2d - gmin) / (gmax - gmin + 1e-9))

    # Simple extrusion: stack the same 2D slice nH times
    nH = max(1, int(round(room_h / (room_d / rows))))
    voxel_grid = np.stack([g] * nH, axis=-1)  # shape (rows, cols, nH)

    launch_3d_viewer(
        voxel_grid,
        room_w=room_w,
        room_d=room_d,
        room_h=room_h,
        output_folder=output_folder,
        title=title,
    )
