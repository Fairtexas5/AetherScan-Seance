# ESP32 CSI Room Mapping — Complete Step-by-Step Guide

> **Hardware ready?** ESP32 flashed with `app_main.c`, connected via USB,
> hotspot ("ashes") active on your phone/router.
> Run **ESP_IDF: Monitor Device** in VS Code to confirm CSI data is streaming.

---

## 📋 Quick Overview

```
ESP32 (USB) ──► Serial Monitor ──► Python Pipeline ──► Heatmap + 3D Charts
```

| Phase | What you do | Time |
|-------|-------------|------|
| **1. Pre-flight** | Install Python deps, start Ollama | 5 min (once) |
| **2. Physical Setup** | Grid tape on floor, plug in ESP32 | 10 min |
| **3. Collect Data** | Walk grid points, run pipeline | 20–60 min |
| **4. Visualize** | Open 3 Plotly HTML charts | 1 min |

---

## Phase 1 — One-Time Setup

### 1.1 Install Python dependencies

```bash
cd /Users/aditsg/csi_hotspot/src
.venv/bin/pip install -r requirements.txt
```

### 1.2 Start Ollama AI (for validation + anomaly explanations)

```bash
# Terminal 1 — keep this running throughout the session
ollama serve
```

If you haven't pulled the model yet:
```bash
ollama pull phi3.5:latest    # ~2.7 GB, one-time download
```

### 1.3 Verify Ollama is working

```bash
ollama run phi3.5:latest "Reply with just: READY"
# Expected: READY
```

---

## Phase 2 — Physical Room Setup

### 2.1 Decide your room grid

> ✅ All dimensions are in **centimetres (cm)**

| Variable | Recommended | Meaning |
|----------|-------------|---------|
| Grid rows | 4–8 | divisions along room depth (front→back) |
| Grid cols | 4–6 | divisions along room width (left→right) |
| Room (WxDxH) | e.g. `500x400x250` | width × depth × height in **cm** |

**Example:** 500cm × 400cm room → 8 rows × 6 cols = 48 grid points (~83cm apart)

### 2.2 Mark the grid on the floor

1. Use masking tape or chalk to mark a **100 cm grid** on the floor.
2. Number rows from **front (0) to back** and columns from **left (0) to right**.
3. Keep the transmitter (your phone hotspot) in **one fixed corner** throughout.

```
        Col 0    Col 1    Col 2    Col 3    Col 4    Col 5
Row 0  [ 0,0 ] [ 0,1 ] [ 0,2 ] [ 0,3 ] [ 0,4 ] [ 0,5 ]  ◄ nearest to hotspot
Row 1  [ 1,0 ] [ 1,1 ] [ 1,2 ] [ 1,3 ] [ 1,4 ] [ 1,5 ]
Row 2  [ 2,0 ] [ 2,1 ] [ 2,2 ] [ 2,3 ] [ 2,4 ] [ 2,5 ]
Row 3  [ 3,0 ] [ 3,1 ] [ 3,2 ] [ 3,3 ] [ 3,4 ] [ 3,5 ]
...
```

### 2.3 Confirm ESP32 is streaming CSI

1. Open VS Code → run **ESP_IDF: Monitor Device**
2. You should see lines like:
   ```
   CSI_DATA,23503655838,-50,64,45,-47,18,0,-13,13,-12,13,...
   ```
3. If you see blank output — check your hotspot "ashes" is ON and the SSID/password in `app_main.c` match.

> ⚠️ **Do NOT leave ESP-IDF Monitor running** when you start the Python pipeline.
> Both can't hold the serial port at the same time. Close it with `Ctrl+X`.

---

## Phase 3 — Quick Check (Serial Monitor)

Before the full pipeline, optionally confirm the serial stream in Python:

```bash
cd /Users/aditsg/csi_hotspot/src
.venv/bin/python3 serial_monitor.py
```

You should see live decoded stats per packet:

```
  #    1  ts=23503655838µs  RSSI= -50dBm  sub= 64  amp= 18.32±3.21  zero=14.3%  [████░░░░░░░░░░░░░░░░]
  #    2  ts=23503758232µs  RSSI= -50dBm  sub= 64  amp= 17.91±3.10  zero=14.3%  [███░░░░░░░░░░░░░░░░░]
```

Press **Ctrl+C** when satisfied, then proceed.

Optionally save a raw capture to file:
```bash
.venv/bin/python3 serial_monitor.py --save output/raw_csi.csv
```

---

## Phase 4 — Collect Room Data (AI Pipeline)

### 4.1 Edit the CONFIG in main.py

Open `src/main.py` and update the `CONFIG` dictionary near line 217:

```python
CONFIG = {
    "grid_rows":         8,      # ← your row count
    "grid_cols":         6,      # ← your col count
    "room_width_m":      5.0,    # ← room width in METRES (500cm ÷ 100)
    "room_height_m":     4.0,    # ← room depth in METRES (400cm ÷ 100)
    "packets_per_point": 200,    # ← packets per grid cell (200 recommended)
    "live_preview":      True,   # ← show partial heatmap after each point
    "auto_retry":        True,   # ← auto-retry bad quality points
    ...
}
```

> 💡 The CONFIG keeps room values in metres internally (e.g. 500cm → 5.0m).
> All output charts, axis labels, and terminal prints will display **centimetres**.

### 4.2 Close ESP-IDF Monitor and run the pipeline

```bash
cd /Users/aditsg/csi_hotspot/src
.venv/bin/python3 main.py
```

### 4.3 Follow the on-screen prompts (per grid point)

The pipeline will walk you through every cell:

```
──────────────────────────────────────
  Point 1/48  →  row=0, col=0
  Place ESP32 at grid position (row=0, col=0)
  Press Enter when ESP32 is in position and steady...
```

**At each prompt:**
1. 🗺️ Carry the ESP32 to the marked grid cell.
2. Stand still for 2–3 seconds.
3. Press **Enter**.
4. Wait ~5–10 seconds while packets are collected.
5. The AI validator gives a **✓ PASS** or **✗ FAIL** verdict.
6. If FAIL → it will automatically retry.
7. Move to the next position.

```
  Collected 200 packets in 5.2s  (38 pkt/s)
  [AI Validator] Checking data quality for r=0 c=0...
  [✓ PASS] Sufficient packets with good RSSI and amplitude variance.
  [AI Anomaly] Analyzing r=0 c=0...
  ✓ NORMAL (confidence: HIGH) → Signal consistent with open floor area.
  [preview] Regenerating heatmap...
```

> 💡 **Tips for good data:**
> - Place the ESP32 flat on the floor or at ~100cm height consistently.
> - Keep people out of the room during collection.
> - Walls and metal furniture cause high-variance readings — totally expected.
> - Each point takes ~10–20s. A 48-point room = ~15–25 min total.

### 4.4 Wait for the final AI report

When all points are collected, the pipeline automatically:
- Generates `output/room_heatmap_final.png`
- Generates `output/room_3d.html` (3D viewer)
- Writes `output/ai_report.txt` (plain-English room interpretation)

---

## Phase 5 — Visualize Results

### 5.1 Generate all three Plotly interactive charts

All dimensions in the command use **centimetres**:

```bash
cd /Users/aditsg/csi_hotspot/src
.venv/bin/python3 main.py --mode visualize --room 500x400x250
```

Or equivalently:
```bash
.venv/bin/python3 pipeline/csi_visualizer.py --data csi_data --room 500x400x250
```

For a different room size — all in cm:
```bash
# 3m × 3m × 2.4m room:
.venv/bin/python3 main.py --mode visualize --room 300x300x240
```

Output:
```
  [2d_heatmap  ]  output/csi_2d_heatmap.html
  [3d_surface  ]  output/csi_3d_surface.html
  [3d_room     ]  output/csi_3d_room.html
```

### 5.2 Open the charts

```bash
open output/csi_2d_heatmap.html   # 2D room grid RF map
open output/csi_3d_surface.html   # 3D surface — drag to rotate
open output/csi_3d_room.html      # 3D room box with measurement spheres
```

### Chart Guide

| Chart | Axes | Best for |
|-------|------|----------|
| **2D Heatmap** | Width (cm) / Depth (cm) | Identifying hot/cold RF zones at a glance |
| **3D Surface** | Width (cm) / Depth (cm) / RF | Understanding signal gradient |
| **3D Room** | Width (cm) / Depth (cm) / Height (cm) | Seeing exactly where each point was taken |

**Colour scale (all charts):**
- 🔵 Dark blue = weak signal / high attenuation (walls, furniture)
- 🟡 Yellow = medium signal
- 🔴 Red = strong signal / clear line-of-sight

---

## Phase 6 — Re-run / Load Saved Session

```bash
# Re-generate charts from previous collection:
.venv/bin/python3 main.py --mode visualize --room 500x400x250

# Convert to .pkl and view with 3D viewer:
.venv/bin/python3 tools/esp32_to_pkl.py --input csi_data --room 5x4x2.5
.venv/bin/python3 main.py --mode load --file esp32_session.pkl

# Simulate (no hardware):
.venv/bin/python3 main.py --mode simulate --room 500x400x250
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No serial port detected` | Unplug/re-plug USB. Check `ls /dev/cu.*` |
| `FAILED TO CONNECT` on ESP32 | Turn on hotspot "ashes", check password in `app_main.c` |
| `CSI: no packets` | ESP32 not connected to hotspot |
| `Permission denied` on serial | Run `sudo chmod 666 /dev/cu.usbserial-*` |
| `[AI offline]` in output | Run `ollama serve` in a separate terminal |
| All packets fail validation | RSSI too low → move hotspot closer |
| Serial port busy | Close VS Code ESP-IDF Monitor (`Ctrl+X`) before running pipeline |

---

## File Structure After Collection

```
src/
├── csi_data/          ← raw per-cell CSI CSV files
│   ├── r00_c00.csv
│   ├── r00_c01.csv
│   └── ...
└── output/
    ├── room_heatmap_final.png   ← matplotlib 2D heatmap  (axes in cm)
    ├── room_3d.html             ← original 3D viewer      (axes in cm)
    ├── csi_2d_heatmap.html      ← Plotly interactive 2D   (axes in cm)
    ├── csi_3d_surface.html      ← Plotly 3D surface        (axes in cm)
    ├── csi_3d_room.html         ← Plotly 3D room structure (axes in cm)
    └── ai_report.txt            ← AI interpretation of the room
```

---

## Command Cheat Sheet

```bash
# One-time setup
ollama serve                                           # keep running

# Monitor ESP32 output (optional pre-check)
.venv/bin/python3 serial_monitor.py

# Run full room mapping pipeline
.venv/bin/python3 main.py

# Generate Plotly charts  (room in cm)
.venv/bin/python3 main.py --mode visualize --room 500x400x250

# Simulate (no hardware, room in cm)
.venv/bin/python3 main.py --mode simulate --room 500x400x250

# Convert data → pkl and view
.venv/bin/python3 tools/esp32_to_pkl.py --input csi_data --room 5x4x2.5
.venv/bin/python3 main.py --mode load --file esp32_session.pkl
```
