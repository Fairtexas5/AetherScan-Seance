"""
plotly_test.py
─────────────────────────────────────────────────────────
WiFi CSI 3D Room Heatmap — CSI-style volumetric view.
A chair sits in the centre of the room.

  Dark purple  = open air (low RF reflection)
  Bright yellow = walls / chair (high RF reflection)

Axes: grid units (integers), no metres labels.
─────────────────────────────────────────────────────────
"""

import numpy as np
import plotly.graph_objects as go

# ── Grid resolution ───────────────────────────────────────────────────
NX, NY, NZ = 30, 30, 15          # voxel counts (X=width, Y=depth, Z=height)

# ── Voxel coordinate arrays ───────────────────────────────────────────
x = np.arange(NX)
y = np.arange(NY)
z = np.arange(NZ)
XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")  # shape (NX,NY,NZ)

# ── Chair position (centre of room, floor-level) ──────────────────────
CX, CY = NX // 2, NY // 2
CHAIR_W   = 4     # seat half-width in grid units
SEAT_Z    = int(NZ * 0.30)      # seat height  (~30% of room height)
BACK_Z_HI = int(NZ * 0.65)     # backrest top (~65%)
BACK_Y    = CY + CHAIR_W        # backrest y edge

# ── RF intensity field ────────────────────────────────────────────────
rf = np.zeros((NX, NY, NZ), dtype=float)

# 1. Wall reflections — bright at boundary voxels
wall_thick = 2
rf[:wall_thick,  :, :] += 1.0     # left
rf[-wall_thick:, :, :] += 1.0     # right
rf[:,  :wall_thick, :] += 1.0     # front
rf[:, -wall_thick:, :] += 1.0     # back
rf[:, :, :wall_thick]  += 0.6     # floor
rf[:, :, -wall_thick:] += 0.4     # ceiling

# 2. Chair seat — solid bright blob at SEAT_Z
seat_mask = (
    (XX >= CX - CHAIR_W) & (XX <= CX + CHAIR_W) &
    (YY >= CY - CHAIR_W) & (YY <= CY + CHAIR_W) &
    (ZZ == SEAT_Z)
)
rf[seat_mask] += 3.5

# 3. Chair backrest — vertical slab at back of seat
back_mask = (
    (XX >= CX - CHAIR_W) & (XX <= CX + CHAIR_W) &
    (YY >= BACK_Y - 1)   & (YY <= BACK_Y + 1) &
    (ZZ >= SEAT_Z)        & (ZZ <= BACK_Z_HI)
)
rf[back_mask] += 3.0

# 4. Chair legs — thin vertical columns at 4 corners
for lx, ly in [
    (CX - CHAIR_W, CY - CHAIR_W),
    (CX + CHAIR_W, CY - CHAIR_W),
    (CX - CHAIR_W, CY + CHAIR_W),
    (CX + CHAIR_W, CY + CHAIR_W),
]:
    leg_mask = (
        (XX >= lx - 1) & (XX <= lx + 1) &
        (YY >= ly - 1) & (YY <= ly + 1) &
        (ZZ <= SEAT_Z)
    )
    rf[leg_mask] += 2.0

# 5. Gaussian fuzz — scattering around the chair body
chair_dist = np.sqrt(
    (XX - CX)**2 + (YY - CY)**2 + (ZZ - SEAT_Z)**2
)
rf += 0.5 * np.exp(-chair_dist**2 / 18.0)

# 6. Open-air centre stays naturally low (no extra boost)

# ── Normalise [0, 1] ─────────────────────────────────────────────────
rf = np.clip(rf, 0, None)
rf = (rf - rf.min()) / (rf.max() - rf.min() + 1e-9)

# ── Flatten for scatter plot ──────────────────────────────────────────
x_flat = XX.ravel()
y_flat = YY.ravel()
z_flat = ZZ.ravel()
v_flat = rf.ravel()

# Show only voxels above a threshold (keeps the plot readable)
thresh = 0.18
mask   = v_flat >= thresh
x_s, y_s, z_s, v_s = x_flat[mask], y_flat[mask], z_flat[mask], v_flat[mask]

# Marker size scales with intensity
msize = 2.5 + v_s * 8

# ── Transmitter marker ────────────────────────────────────────────────
tx_scatter = go.Scatter3d(
    x=[1], y=[1], z=[int(NZ * 0.4)],
    mode="markers+text",
    marker=dict(size=14, color="lime", symbol="diamond",
                line=dict(color="white", width=1)),
    text=["TX"],
    textposition="top center",
    textfont=dict(color="lime", size=13),
    name="Transmitter",
)

# ── Main heatmap scatter ──────────────────────────────────────────────
heatmap_scatter = go.Scatter3d(
    x=x_s, y=y_s, z=z_s,
    mode="markers",
    marker=dict(
        size=msize,
        color=v_s,
        colorscale="plasma",
        opacity=0.55,
        cmin=0, cmax=1,
        colorbar=dict(
            title=dict(text="RF Intensity", font=dict(color="white", size=13)),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Low", "", "Mid", "", "High"],
            tickfont=dict(color="white"),
            x=1.02,
        ),
    ),
    text=[
        f"RF = {v:.2f}"
        for v in v_s
    ],
    hovertemplate="X=%{x}  Y=%{y}  Z=%{z}<br>%{text}<extra></extra>",
    name="CSI RF Field",
)

# ── Assemble figure ───────────────────────────────────────────────────
fig = go.Figure(data=[heatmap_scatter, tx_scatter])

fig.update_layout(
    title=dict(
        text="WiFi CSI 3D Room Map — Chair in Centre",
        font=dict(size=17, color="white"),
        x=0.5, xanchor="center",
    ),
    scene=dict(
        xaxis=dict(
            title=dict(text="X", font=dict(color="white")),
            tickvals=list(range(0, NX + 1, 5)),
            backgroundcolor="#0d0d1a",
            gridcolor="#2a2a44",
            showbackground=True,
            tickfont=dict(color="white"),
        ),
        yaxis=dict(
            title=dict(text="Y", font=dict(color="white")),
            tickvals=list(range(0, NY + 1, 5)),
            backgroundcolor="#0d0d1a",
            gridcolor="#2a2a44",
            showbackground=True,
            tickfont=dict(color="white"),
        ),
        zaxis=dict(
            title=dict(text="Z", font=dict(color="white")),
            tickvals=list(range(0, NZ + 1, 3)),
            backgroundcolor="#0d0d1a",
            gridcolor="#2a2a44",
            showbackground=True,
            tickfont=dict(color="white"),
        ),
        bgcolor="#07070f",
        camera=dict(eye=dict(x=1.7, y=-1.7, z=1.1)),
        aspectratio=dict(x=1, y=1, z=0.55),
        annotations=[
            dict(
                x=CX, y=CY, z=BACK_Z_HI + 1,
                text="Chair", showarrow=False,
                font=dict(size=13, color="#ffccff"),
                xanchor="center",
            ),
        ],
    ),
    paper_bgcolor="#07070f",
    font=dict(color="white", family="Arial"),
    legend=dict(
        bgcolor="#12122a", bordercolor="#444",
        x=0.01, y=0.99,
        font=dict(size=11, color="white"),
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    width=980, height=660,
)

fig.show()
print("CSI heatmap rendered — browser tab opened.")
