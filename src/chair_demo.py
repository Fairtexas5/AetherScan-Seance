"""
chair_demo.py
─────────────────────────────────────────────────────────
Demo CSI heatmaps from the chair photo grid.

Grid observed in photos:
  6 rows (0–5) × 4 cols (0–3)  — each cell ≈ 30cm (floor tile)
  [0,0] = top-left corner (front of chair, nearest hotspot)

Chair position (from front photo):
  The chair sits roughly above rows 1–3, cols 0–2
  • Chair back    ≈ row 1,  cols 1–2  (high metal reflection)
  • Chair seat    ≈ row 2,  cols 0–2  (very high – thick metal)
  • 5-star base   ≈ row 2–3, cols 0–3 (medium–high)
  • RF shadow     ≈ rows 4–5          (attenuated – occluded)

Run:
  python src/chair_demo.py
─────────────────────────────────────────────────────────
"""

import os
import math
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Output folder ─────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CSI variance grid (from photo analysis) ───────────────────────────
CSI_VARIANCE = np.array([
    # Col:  0      1      2      3
    [0.45,  0.50,  0.48,  0.38],   # Row 0 — clear LOS to TX, near chair back
    [0.58,  0.88,  0.82,  0.44],   # Row 1 — chair arms + back (metal peaks)
    [0.62,  0.95,  0.91,  0.47],   # Row 2 — chair SEAT centre (highest variance)
    [0.55,  0.78,  0.72,  0.42],   # Row 3 — 5-star base legs (metal)
    [0.30,  0.24,  0.28,  0.36],   # Row 4 — RF shadow behind chair
    [0.25,  0.20,  0.22,  0.32],   # Row 5 — deep shadow + far from TX
], dtype=float)

GRID_ROWS, GRID_COLS = CSI_VARIANCE.shape   # 6, 4
CELL_CM = 30                                 # one floor tile = 30 cm

X_CM     = [c * CELL_CM for c in range(GRID_COLS)]
Y_CM     = [r * CELL_CM for r in range(GRID_ROWS)]
X_LABELS = [f"{v}cm" for v in X_CM]
Y_LABELS = [f"{v}cm" for v in Y_CM]

# ── Common heatmap colorscale ─────────────────────────────────────────
CS = [
    [0.00, "#06003c"],
    [0.25, "#1a0099"],
    [0.45, "#0044ff"],
    [0.62, "#00cccc"],
    [0.80, "#ffee00"],
    [1.00, "#ff2200"],
]

# ── Dark theme ────────────────────────────────────────────────────────
DARK_BG  = "#0d0d1a"
PAPER_BG = "#12122a"
FONT_COL = "white"
FONT_FAM = "Inter, Arial, sans-serif"


# ── Colour mapping helper ─────────────────────────────────────────────
def val_to_hex(v, vmin=0.20, vmax=0.95):
    """Map a CSI variance value to a hex colour using the heatmap scale."""
    t = max(0.0, min(1.0, (v - vmin) / max(vmax - vmin, 1e-6)))
    for i in range(len(CS) - 1):
        t0, c0 = CS[i]
        t1, c1 = CS[i + 1]
        if t0 <= t <= t1 + 1e-9:
            f  = (t - t0) / max(t1 - t0, 1e-9)
            r0, g0, b0 = int(c0[1:3],16), int(c0[3:5],16), int(c0[5:7],16)
            r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
            return f"#{int(r0+f*(r1-r0)):02x}{int(g0+f*(g1-g0)):02x}{int(b0+f*(b1-b0)):02x}"
    return CS[-1][1]


def solid_box(x0, x1, y0, y1, z0, z1, colour, name, opacity=0.82, show=True):
    """Return a solid-coloured Mesh3d rectangular box."""
    xs = [x0, x1, x1, x0,  x0, x1, x1, x0]
    ys = [y0, y0, y1, y1,  y0, y0, y1, y1]
    zs = [z0, z0, z0, z0,  z1, z1, z1, z1]
    i  = [0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6]
    j  = [1, 3, 2, 5, 3, 6, 5, 7, 6, 1, 7, 2]
    k  = [2, 2, 5, 6, 6, 7, 7, 6, 1, 2, 2, 3]
    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i, j=j, k=k,
        color=colour,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
        showlegend=show,
        flatshading=True,
        lighting=dict(ambient=0.65, diffuse=0.85, specular=0.3),
    )


# ══════════════════════════════════════════════════════════════════════
#  CHART 1 — 2D Interactive Heatmap
# ══════════════════════════════════════════════════════════════════════
def chart_2d():
    fig = px.imshow(
        CSI_VARIANCE,
        labels={"x": "Width (cm)", "y": "Depth (cm)", "color": "CSI Variance"},
        x=X_LABELS,
        y=Y_LABELS,
        color_continuous_scale=CS,
        zmin=0.0, zmax=1.0,
        aspect="auto",
        title="CSI 2D RF Heatmap — Chair Obstacle (6×4 Grid)",
        text_auto=".2f",
    )
    fig.add_shape(type="rect", x0=-0.5, x1=2.5, y0=0.5, y1=3.5,
                  line=dict(color="white", width=2, dash="dash"),
                  fillcolor="rgba(255,255,255,0.04)")
    fig.add_annotation(x=1.0, y=0.85, text="🪑 Chair", showarrow=False,
                       font=dict(color="white", size=13, family=FONT_FAM),
                       bgcolor="rgba(0,0,0,0.55)")
    fig.add_annotation(x=1.5, y=-0.45, text="📡 TX (hotspot)", showarrow=False,
                       font=dict(color="#aaffaa", size=11, family=FONT_FAM),
                       bgcolor="rgba(0,40,0,0.6)")
    fig.update_layout(
        title_font_size=18, title_x=0.5,
        paper_bgcolor=PAPER_BG, plot_bgcolor=DARK_BG,
        font=dict(color=FONT_COL, family=FONT_FAM),
        margin=dict(l=60, r=60, t=80, b=80),
        coloraxis_colorbar=dict(
            title="RF Intensity",
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Low", "", "Mid", "", "High"],
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, side="top", tickfont=dict(size=13))
    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed", tickfont=dict(size=13))
    path = os.path.join(OUTPUT_DIR, "chair_2d_heatmap.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[chair_demo] 2D heatmap  → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
#  CHART 2 — 3D Surface Heatmap
# ══════════════════════════════════════════════════════════════════════
def chart_3d_surface():
    x_cm = np.array(X_CM, dtype=float)
    y_cm = np.array(Y_CM, dtype=float)
    X, Y = np.meshgrid(x_cm, y_cm)
    Z    = CSI_VARIANCE.copy()

    surf = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=CS,
        cmin=0, cmax=1,
        colorbar=dict(
            title=dict(text="RF Intensity", side="right"),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Low", "", "Mid", "", "High"],
            len=0.7,
        ),
        hovertemplate=(
            "Col: %{x:.0f}cm<br>"
            "Row: %{y:.0f}cm<br>"
            "RF Variance: %{z:.3f}<extra></extra>"
        ),
        lighting=dict(ambient=0.6, diffuse=0.9, specular=0.4, roughness=0.4),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white",
                             project=dict(z=True))),
    )
    peak_r, peak_c = 2, 1
    peak_marker = go.Scatter3d(
        x=[X_CM[peak_c]], y=[Y_CM[peak_r]], z=[CSI_VARIANCE[peak_r, peak_c] + 0.05],
        mode="markers+text",
        marker=dict(size=6, color="white", symbol="diamond"),
        text=["🪑 Chair seat peak"],
        textfont=dict(color="white", size=11),
        textposition="top center",
        hoverinfo="skip", showlegend=False,
    )
    fig = go.Figure(data=[surf, peak_marker])
    fig.update_layout(
        title=dict(text="CSI 3D Surface Heatmap — Chair Obstacle",
                   x=0.5, font=dict(size=18, color=FONT_COL)),
        scene=dict(
            xaxis=dict(title="Width (cm)", backgroundcolor=DARK_BG,
                       gridcolor="#333", showbackground=True, range=[0, 90]),
            yaxis=dict(title="Depth (cm)", backgroundcolor=DARK_BG,
                       gridcolor="#333", showbackground=True, range=[0, 150]),
            zaxis=dict(title="RF Variance", backgroundcolor=DARK_BG,
                       gridcolor="#333", showbackground=True, range=[0, 1.1]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
            bgcolor=DARK_BG,
        ),
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COL, family=FONT_FAM),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    path = os.path.join(OUTPUT_DIR, "chair_3d_surface.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[chair_demo] 3D surface  → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
#  CHART 3 — 3D Room Structure
#  FIXES vs previous version:
#    ✓ Measurement points at FLOOR level (z=5cm) — ESP32 on tiles
#    ✓ Chair correctly above floor: seat at 42–50cm, back at 50–122cm
#    ✓ Chair parts are solid Mesh3d coloured with the heatmap scale
# ══════════════════════════════════════════════════════════════════════
def chart_3d_room():
    ROOM_H_CM = 240

    # ── Chair geometry — centred over rows 1–3, cols 0–2 ──────────────
    # Room x = col * 30cm, y = row * 30cm
    # Chair occupies: x: 5–70cm (cols 0–2), y: 30–90cm (rows 1–3)
    CX0, CX1 = 5,  70     # left/right extent
    CY0, CY1 = 30, 90     # front/back extent (rows 1–3)
    MX = (CX0 + CX1) / 2  # midX = 37.5
    MY = (CY0 + CY1) / 2  # midY = 60

    # Part heights
    SEAT_Z0, SEAT_Z1 = 42, 50   # seat slab
    BACK_Z0, BACK_Z1 = 50, 122  # backrest
    COL_Z0,  COL_Z1  = 0,  42   # central column (pedestal)

    # Heatmap-matched colours per part
    seat_col  = val_to_hex(0.95)  # red    — hottest RF (seat)
    back_col  = val_to_hex(0.86)  # orange — chair back
    arm_col   = val_to_hex(0.79)  # yellow-orange — arms
    col_col   = val_to_hex(0.70)  # yellow — column
    leg_col   = val_to_hex(0.60)  # cyan-yellow — legs

    chair_parts = [
        # Seat
        solid_box(CX0, CX1, CY0, CY1, SEAT_Z0, SEAT_Z1,
                  seat_col, "Chair seat", opacity=0.90),
        # Backrest (thin slab at front of seat)
        solid_box(CX0+6, CX1-6, CY0, CY0+7, BACK_Z0, BACK_Z1,
                  back_col, "Chair backrest", opacity=0.85),
        # Left arm
        solid_box(CX0, CX0+7, CY0+5, CY1-5, SEAT_Z1, SEAT_Z1+48,
                  arm_col, "Chair arm L", opacity=0.78),
        # Right arm
        solid_box(CX1-7, CX1, CY0+5, CY1-5, SEAT_Z1, SEAT_Z1+48,
                  arm_col, "Chair arm R", opacity=0.78, show=False),
        # Central column / pedestal
        solid_box(MX-5, MX+5, MY-5, MY+5, COL_Z0, COL_Z1,
                  col_col, "Chair column", opacity=0.85),
    ]
    # 5-star base legs (5 spokes, low profile)
    leg_len = 24
    for a in range(0, 360, 72):
        rx = math.cos(math.radians(a)) * leg_len
        ry = math.sin(math.radians(a)) * leg_len
        lx0 = min(MX, MX + rx) - 3
        lx1 = max(MX, MX + rx) + 3
        ly0 = min(MY, MY + ry) - 3
        ly1 = max(MY, MY + ry) + 3
        chair_parts.append(
            solid_box(lx0, lx1, ly0, ly1, 0, 5,
                      leg_col, "_leg", opacity=0.75, show=False)
        )

    # ── Room wireframe ─────────────────────────────────────────────────
    rw, rd, rh = 90, 150, ROOM_H_CM
    wpts = [
        (0,0,0),(rw,0,0),(rw,rd,0),(0,rd,0),(0,0,0),
        (0,0,rh),(rw,0,rh),(rw,rd,rh),(0,rd,rh),(0,0,rh),
        None,(rw,0,0),(rw,0,rh),
        None,(rw,rd,0),(rw,rd,rh),
        None,(0,rd,0),(0,rd,rh),
    ]
    wx = [p[0] if p else None for p in wpts]
    wy = [p[1] if p else None for p in wpts]
    wz = [p[2] if p else None for p in wpts]
    wireframe = go.Scatter3d(
        x=wx, y=wy, z=wz, mode="lines",
        line=dict(color="#4488ff", width=2),
        name="Room boundary", hoverinfo="skip"
    )

    # ── Floor grid lines ───────────────────────────────────────────────
    grid_lines = []
    for c in range(GRID_COLS + 1):
        x = c * CELL_CM
        grid_lines.append(go.Scatter3d(
            x=[x, x], y=[0, GRID_ROWS * CELL_CM], z=[0, 0],
            mode="lines", line=dict(color="rgba(80,80,200,0.25)", width=1),
            showlegend=False, hoverinfo="skip"))
    for r in range(GRID_ROWS + 1):
        y = r * CELL_CM
        grid_lines.append(go.Scatter3d(
            x=[0, GRID_COLS * CELL_CM], y=[y, y], z=[0, 0],
            mode="lines", line=dict(color="rgba(80,80,200,0.25)", width=1),
            showlegend=False, hoverinfo="skip"))

    # ── Measurement points — ON THE FLOOR (z=5cm) ─────────────────────
    MEAS_Z = 5     # ESP32 placed on floor tiles
    pt_x, pt_y, pt_z, pt_val, pt_text = [], [], [], [], []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            pt_x.append(float(X_CM[c]))
            pt_y.append(float(Y_CM[r]))
            pt_z.append(float(MEAS_Z))
            val = float(CSI_VARIANCE[r, c])
            pt_val.append(val)
            icon = "🔴" if val > 0.7 else ("🔵" if val < 0.32 else "🟡")
            pt_text.append(
                f"<b>Grid [{r},{c}]</b><br>"
                f"({X_CM[c]}cm, {Y_CM[r]}cm)<br>"
                f"RF: {val:.3f}  {icon}"
            )

    vmin, vmax = min(pt_val), max(pt_val)
    pt_norm = [(v - vmin) / max(vmax - vmin, 1e-9) for v in pt_val]

    # Vertical RF bars from floor
    bar_traces = []
    for px_, py_, pv in zip(pt_x, pt_y, pt_val):
        bar_h = pv * 38
        bar_traces.append(go.Scatter3d(
            x=[px_, px_], y=[py_, py_], z=[MEAS_Z, MEAS_Z + bar_h],
            mode="lines",
            line=dict(color="rgba(120,210,255,0.28)", width=4),
            showlegend=False, hoverinfo="skip"))

    points = go.Scatter3d(
        x=pt_x, y=pt_y, z=pt_z,
        mode="markers",
        marker=dict(
            size=11,
            color=pt_norm,
            colorscale=CS,
            cmin=0, cmax=1,
            opacity=0.95,
            colorbar=dict(
                title=dict(text="RF Intensity", side="right"),
                tickvals=[0, 0.5, 1.0],
                ticktext=["Low", "Mid", "High"],
                x=1.02, len=0.55, thickness=15,
            ),
        ),
        text=pt_text,
        hovertemplate="%{text}<extra></extra>",
        name="Measurement points (floor)",
    )

    # Grid cell labels
    labels = go.Scatter3d(
        x=pt_x, y=pt_y, z=[MEAS_Z + 14] * len(pt_x),
        mode="text",
        text=[f"[{r},{c}]" for r in range(GRID_ROWS) for c in range(GRID_COLS)],
        textfont=dict(size=8, color="rgba(200,200,200,0.65)"),
        showlegend=False, hoverinfo="skip",
    )

    # TX marker
    tx = go.Scatter3d(
        x=[45], y=[-10], z=[120],
        mode="markers+text",
        marker=dict(size=10, color="#00ff88", symbol="diamond"),
        text=["📡 TX"],
        textfont=dict(color="#00ff88", size=12),
        textposition="top center",
        name="Transmitter (hotspot)",
        hoverinfo="skip",
    )

    all_traces = (
        [wireframe]
        + grid_lines
        + bar_traces
        + [points, labels]
        + chair_parts
        + [tx]
    )

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(
            text="CSI 3D Room Structure — Chair Position Mapped",
            x=0.5, font=dict(size=18, color=FONT_COL)
        ),
        scene=dict(
            xaxis=dict(title="Width (0–90cm)", range=[-5, 98],
                       backgroundcolor=DARK_BG, gridcolor="#222", showbackground=True),
            yaxis=dict(title="Depth (0–150cm)", range=[-18, 160],
                       backgroundcolor=DARK_BG, gridcolor="#222", showbackground=True),
            zaxis=dict(title="Height (0–240cm)", range=[0, 250],
                       backgroundcolor=DARK_BG, gridcolor="#222", showbackground=True),
            aspectmode="manual",
            aspectratio=dict(x=0.9, y=1.5, z=2.2),
            camera=dict(eye=dict(x=1.8, y=-1.9, z=0.85)),
            bgcolor=DARK_BG,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.55)", font=dict(color=FONT_COL, size=11),
                    x=0.01, y=0.99),
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COL, family=FONT_FAM),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    path = os.path.join(OUTPUT_DIR, "chair_3d_room.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[chair_demo] 3D room     → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
#  RUN ALL
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Chair CSI Demo Visualizer")
    print("  Grid: 6 rows × 4 cols  (30cm per cell)")
    print("  Chair at approx rows 1–3, cols 0–2")
    print("=" * 60)
    print()

    p2d  = chart_2d()
    p3ds = chart_3d_surface()
    p3dr = chart_3d_room()

    print()
    print("=" * 60)
    print("  Opening charts in browser...")
    print("=" * 60)

    import webbrowser, pathlib
    for p in [p2d, p3ds, p3dr]:
        webbrowser.open(pathlib.Path(os.path.abspath(p)).as_uri())
