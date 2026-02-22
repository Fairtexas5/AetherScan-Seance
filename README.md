# AetherScan — Séance

**WiFi Channel State Information (CSI) Room Mapper with AI-Powered Analysis**

AetherScan Séance turns an ESP32 microcontroller and a standard WiFi hotspot into a room-scale RF sensing system. It captures raw CSI data from ambient WiFi beacon frames, processes IQ subcarrier values across a physical floor grid, and uses a local LLM (via Ollama) to validate, detect anomalies, and produce a plain-English interpretation of the room's radio environment. The results are rendered as interactive 2D and 3D Plotly visualizations.

---

## Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Usage](#usage)
- [Pipeline Deep Dive](#pipeline-deep-dive)
- [AI Layer](#ai-layer)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Understanding Your Results](#understanding-your-results)

---

## How It Works

1. A **phone hotspot** (SSID: `ashes`) broadcasts 2.4 GHz WiFi beacon frames roughly every 100 ms.
2. An **ESP32** connected to the hotspot captures 64-subcarrier IQ CSI data from each beacon using its built-in hardware CSI extraction feature, streaming the raw data over USB serial at 115200 baud.
3. A **Python pipeline** reads the serial stream, parses IQ pairs (amplitude = √(I² + Q²)), and stores packets per grid cell as CSV files.
4. After data collection, a **LangGraph state machine** validates each data point using a local LLM, detects anomalies by comparing cells, and generates a final room report.
5. The pipeline produces **Plotly HTML charts** (2D heatmap, 3D surface, 3D room structure) and a **Matplotlib PNG heatmap**.

The key insight is that objects, walls, and furniture all affect how WiFi signals propagate — their presence can be inferred from the amplitude and phase patterns of CSI subcarriers.

---

## System Architecture

```
Phone Hotspot (TX)
       │  2.4 GHz beacon frames
       ▼
   ESP32 (RX)
       │  CSI_DATA lines @ 115200 baud over USB
       ▼
  MacBook Pro
       │  pyserial
       ▼
Python Pipeline (src/)
       │  LangChain + LangGraph
       ▼
  Ollama AI (phi3.5:latest)  ←→  Validation / Anomaly / Report
       │
       ▼
  Output Charts (output/)
```

**Serial data format emitted by the ESP32:**
```
CSI_DATA, {timestamp_µs}, {rssi_dBm}, {n_sub}, {extra1}, {extra2}, {I0},{Q0},{I1},{Q1},...
  Field 0      Field 1        Field 2    Field 3   Field 4   Field 5   Field 6 onwards
```
- `n_sub` = 64 subcarriers → 128 IQ values per packet
- IQ pairs parsed at field index 6 by `pipeline/csi_parser.py`

---

## Hardware Requirements

| Component | Purpose | Notes |
|-----------|---------|-------|
| **ESP32 Dev Board** | CSI receiver | Any standard ESP32 with USB |
| **Smartphone** (or router) | WiFi transmitter / hotspot | SSID: `ashes`, password: `987654321` |
| **MacBook Pro** | Host computer | macOS; USB serial connection to ESP32 |
| **USB cable** | Data + power for ESP32 | Micro-USB or USB-C depending on board |
| **Masking tape** | Floor grid marking | For physical grid layout |

> **Note:** The firmware is configured for a hotspot named `ashes` with password `987654321`. These can be changed in `main/app_main.c` (`WIFI_SSID` / `WIFI_PASS` defines).

---

## Software Requirements

### ESP32 Firmware
- [VS Code](https://code.visualstudio.com/) with the [ESP-IDF extension](https://marketplace.visualstudio.com/items?itemName=espressif.esp-idf-vscode-extension)
- ESP-IDF v5.x (configured via the extension)

### Python (Host Machine)
- Python 3.10 or newer
- All dependencies listed in `src/requirements.txt`:

| Package | Purpose |
|---------|---------|
| `langgraph >= 0.2.0` | Pipeline state machine orchestration |
| `langchain-ollama >= 0.1.0` | LLM client for local Ollama |
| `langchain-core >= 0.2.0` | LangChain core abstractions |
| `pyserial >= 3.5` | USB serial communication with ESP32 |
| `numpy >= 1.24.0` | Signal processing and array math |
| `scipy >= 1.10.0` | Gaussian smoothing, edge detection |
| `matplotlib >= 3.7.0` | 2D heatmap PNG generation |
| `pandas >= 2.0.0` | CSV I/O and data manipulation |
| `plotly >= 5.18.0` | Interactive 3D/2D HTML visualizations |
| `pyshark >= 0.6.0` | macOS WiFi capture via Wireshark/tshark |

### AI Backend
- [Ollama](https://ollama.com/download) — local LLM server (no API key required)
- Model: `phi3.5:latest` (~2.7 GB download, Microsoft's efficient instruction-following model)
- Runs fully on-device via CPU or Apple Metal — no cloud calls

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/AetherScan-Seance.git
cd AetherScan-Seance
```

### 2. Set up the Python virtual environment
```bash
cd src
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. Install and configure Ollama
```bash
# Install from https://ollama.com/download, then:
ollama pull phi3.5:latest    # ~2.7 GB one-time download
ollama serve                 # Start the local AI server (keep running)
```

Verify it works:
```bash
ollama run phi3.5:latest "Reply with just: READY"
# Expected output: READY
```

### 4. (Optional) Install Wireshark for capture mode
```bash
brew install --cask wireshark
```

### 5. Flash the ESP32 firmware
Open the project root in VS Code (where `CMakeLists.txt` lives):

1. Press `Ctrl+Shift+P` → **ESP-IDF: Select Port to Use** → choose your ESP32's COM port
2. Press `Ctrl+Shift+P` → **ESP-IDF: SDK Configuration Editor** → search `CSI` → enable **WiFi CSI** under *Component config → WiFi* → Save
3. Press `Ctrl+Shift+P` → **ESP-IDF: Build your project** (2–5 min)
4. Press `Ctrl+Shift+P` → **ESP-IDF: Flash your project**
5. Press `Ctrl+Shift+P` → **ESP-IDF: Monitor your device** — you should see `CSI_DATA,...` lines scrolling

> ⚠️ **Before running the Python pipeline**, close the ESP-IDF Monitor with `Ctrl+X`. Two processes cannot hold the serial port simultaneously.

---

## Hardware Setup

### Room Grid Layout

1. Measure your room dimensions (width × depth × height in centimetres).
2. Use masking tape to mark a grid on the floor. Recommended starting spacing: **100 cm**.
3. Label each point `r0c0`, `r0c1`, etc. — rows run front-to-back, columns run left-to-right.
4. Place the phone hotspot in one corner (the transmitter, TX). **Do not move it during the entire session.**
5. Elevate the hotspot to approximately 100 cm height.

```
        Col 0    Col 1    Col 2    Col 3
Row 0  [ 0,0 ] [ 0,1 ] [ 0,2 ] [ 0,3 ]   ← nearest to hotspot (TX)
Row 1  [ 1,0 ] [ 1,1 ] [ 1,2 ] [ 1,3 ]
Row 2  [ 2,0 ] [ 2,1 ] [ 2,2 ] [ 2,3 ]
Row 3  [ 3,0 ] [ 3,1 ] [ 3,2 ] [ 3,3 ]
```

**Grid spacing guide:**

| Spacing | Points (5×4m room) | Quality | Time |
|---------|-------------------|---------|------|
| 100 cm | 5 × 4 = 20 | Good for first run | ~10 min |
| 50 cm | 10 × 8 = 80 | Detailed | ~40 min |
| 25 cm | 20 × 16 = 320 | Research quality | ~2.5 hr |

---

## Usage

All commands are run from the `src/` directory with the virtual environment activated.

### Quick hardware check (optional but recommended)
```bash
.venv/bin/python3 serial_monitor.py
```
You should see live decoded CSI stats. Press `Ctrl+C` when satisfied.

### Mode 1: Full AI pipeline (requires ESP32 + Ollama)

Edit `CONFIG` near line 217 of `src/main.py` to match your room:
```python
CONFIG = {
    "grid_rows":         8,      # divisions front-to-back
    "grid_cols":         6,      # divisions left-to-right
    "room_width_m":      5.0,    # room width in metres
    "room_height_m":     4.0,    # room depth in metres
    "packets_per_point": 200,    # CSI frames per grid cell
    "live_preview":      True,
    "auto_retry":        True,
}
```
Then run:
```bash
# Terminal 1 — keep Ollama running
ollama serve

# Terminal 2 — start collection
.venv/bin/python3 main.py
```
At each prompt, carry the ESP32 to the marked grid cell, stand still, and press Enter. The AI validator will confirm quality, detect anomalies, and show a live heatmap preview after each point.

### Mode 2: Simulate (no hardware needed)
```bash
# Room dimensions in centimetres: WxDxH
.venv/bin/python3 main.py --mode simulate --room 500x400x250
```
Generates physically accurate synthetic CSI data using free-space path loss, wall reflections, and random furniture blobs, then launches the 3D viewer.

### Mode 3: Capture via Wireshark (macOS only)
```bash
# Passive WiFi sniffing via tshark — requires sudo
sudo .venv/bin/python3 main.py --mode capture --interface en0 --duration 60 --room 500x400x250
```

### Mode 4: Load a saved session
```bash
.venv/bin/python3 main.py --mode load --file esp32_session.pkl
```

### Mode 5: Generate Plotly charts from existing collected data
```bash
# All dimensions in centimetres
.venv/bin/python3 main.py --mode visualize --room 500x400x250 --data csi_data
```

### Utility tools
```bash
# Convert collected CSV files to .pkl for the 3D viewer (room dims in metres)
.venv/bin/python3 tools/esp32_to_pkl.py --input csi_data --room 5x4x2.5

# Guided walk-around capture with position prompts
.venv/bin/python3 tools/collect_positions.py --rows 4 --cols 5 --room 5x4x2.5 --interface en0
```

---

## Pipeline Deep Dive

The main collection mode runs a **LangGraph state machine** with the following nodes:

```
node_setup → node_collect_point → node_ai_validate_point
                  ↑  (retry < 3)          │ FAIL
                  └──────────────────────┘
                                          │ PASS
                                          ▼
                              node_ai_anomaly_detection
                                          │
                                          ▼
                              node_live_preview
                                          │
                                          ▼
                              node_advance (save CSV, increment row/col)
                                          │
                              ┌───────────┴───────────┐
                              │ more points            │ done
                              ▼                        ▼
                    (loop back to collect)   node_process_heatmap
                                                       │
                                                       ▼
                                           node_ai_interpret_heatmap
                                                       │
                                                       ▼
                                                      END
```

**CSI Parsing (csi_parser.py):**

Each raw serial line is validated (`serial_utils.py`) to confirm it starts with `CSI_DATA`, has ≥ 8 fields, and contains a numeric RSSI. Valid lines are parsed into a `ParsedResult` dict with:
- `timestamp_us` — ESP32 hardware timer value
- `rssi` — received signal strength in dBm
- `n_sub` — number of subcarriers (64 for this firmware)
- `iq_pairs` — list of (I, Q) tuples
- `amplitude` — numpy array of √(I² + Q²) per subcarrier

---

## AI Layer

All AI inference runs **locally** via Ollama. No API keys or internet connection required.

**Model:** `phi3.5:latest` (Microsoft, ~2.7 GB)
**Server:** `localhost:11434`
**Expected inference times on MacBook Pro:** 5–40 seconds per call depending on task

Three AI components:

| Module | Purpose | Output |
|--------|---------|--------|
| `ai_validator.py` | Statistical + LLM quality check per grid point. Checks ≥50 packets, RSSI in range, amplitude variance. Re-asks up to 3× if uncertain. | `PASS` / `FAIL` |
| `ai_anomaly.py` | Compares current point stats to all prior points. Asks the LLM if the reading is unusual and what might cause it. | `NORMAL` or `ANOMALOUS` + explanation sentence |
| `ai_interpreter.py` | Final room-level analysis. Computes grid statistics, summarises anomalies, asks the LLM to describe the room's RF zones and suggest 2 improvement tips. | Plain-English report → `ai_report.txt` |

Configuration is in `pipeline/ai_client.py`:
```python
MODEL_NAME   = "phi3.5:latest"
OLLAMA_URL   = "127.0.0.1:11434"
TEMPERATURE  = 0.1      # Low = focused, factual responses
MAX_TOKENS   = 512
NUM_THREADS  = 6        # Tune to your CPU core count
```

---

## Output Files

After a complete pipeline run, the following files are written to `src/output/`:

| File | Description |
|------|-------------|
| `room_heatmap_final.png` | Matplotlib 2D heatmap with Gaussian smoothing and edge detection, axes in cm |
| `room_3d.html` | Plotly 3D voxel viewer — shows signal intensity through the room volume |
| `csi_2d_heatmap.html` | Interactive 2D Plotly heatmap — hover for exact RF values at each grid cell |
| `csi_3d_surface.html` | 3D surface plot of signal strength — drag to rotate |
| `csi_3d_room.html` | 3D room bounding box with measurement spheres showing exact capture positions |
| `ai_report.txt` | Plain-English LLM interpretation: RF zones, anomalies, improvement suggestions |

Raw per-cell CSI data is saved to `src/csi_data/r{row}_c{col}.csv` during collection.

---

## Project Structure

```
AetherScan-Seance/
├── main/
│   ├── app_main.c          ← ESP32 firmware (C): connects to hotspot, streams CSI
│   └── CMakeLists.txt      ← ESP-IDF component registration
├── CMakeLists.txt          ← ESP-IDF project root
├── sample.txt              ← Example raw CSI_DATA serial output
│
└── src/
    ├── main.py             ← Entry point — 5 run modes, argument parser, CONFIG dict
    ├── serial_monitor.py   ← Standalone live ESP32 stream viewer with decoded stats
    ├── test_pipeline.py    ← End-to-end test (no hardware required)
    ├── plotly_test.py      ← Standalone Plotly rendering test
    ├── chair_demo.py       ← Demo: maps signal around a single chair object
    ├── requirements.txt    ← Python dependencies
    │
    ├── pipeline/
    │   ├── csi_parser.py      ← IQ parser: single source of truth for line parsing
    │   ├── serial_utils.py    ← Auto-detect serial port, validate + read lines
    │   ├── nodes.py           ← LangGraph node function implementations
    │   ├── graph.py           ← LangGraph state machine definition and wiring
    │   ├── validator.py       ← Statistical CSI quality checks (packet count, RSSI, variance)
    │   ├── ai_validator.py    ← LLM-based data quality check (phi3.5)
    │   ├── ai_anomaly.py      ← LLM anomaly detection + explanation per grid point
    │   ├── ai_interpreter.py  ← Final LLM room report generation
    │   ├── ai_client.py       ← Ollama client configuration
    │   ├── heatmap.py         ← Matplotlib 2D heatmap with smoothing and edge detection
    │   ├── csi_visualizer.py  ← Plotly 2D heatmap, 3D surface, 3D room structure charts
    │   ├── visualizer_3d.py   ← Plotly 3D voxel viewer (animated slice view)
    │   ├── simulator.py       ← Synthetic CSI data: FSPL model + wall reflections + furniture
    │   └── capture_macos.py   ← WiFi packet capture via Wireshark/tshark (pyshark)
    │
    ├── tools/
    │   ├── collect_positions.py  ← Guided walk-around capture with per-position prompts
    │   └── esp32_to_pkl.py       ← Converts csi_data/ CSVs to .pkl for the 3D viewer
    │
    ├── output/                   ← All generated charts and reports land here
    └── test_csi_data/            ← Sample 3×3 grid CSV files for testing (r00_c00.csv etc.)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No serial port detected` | Unplug and re-plug USB. Run `ls /dev/cu.*` to see available ports. |
| ESP32 shows `FAILED TO CONNECT` | Turn on phone hotspot `ashes`. Verify SSID/password in `main/app_main.c`. |
| `CSI: no packets` / no `CSI_DATA` lines | ESP32 is not connected to the hotspot. Check hotspot is on and in range. |
| `Permission denied` on serial port | Run `sudo chmod 666 /dev/cu.usbserial-*` |
| `[AI offline]` appears in output | Start Ollama: `ollama serve` in a separate terminal. |
| All points fail validation | RSSI too low — move the hotspot closer, or check hotspot is on. |
| Serial port busy error | Close VS Code ESP-IDF Monitor (`Ctrl+X`) before running the Python pipeline. |
| `pyshark` / capture mode fails | Install Wireshark: `brew install --cask wireshark`. Run with `sudo`. |

---

## Understanding Your Results

### Heatmap color scale
| Color | RF Intensity | Physical meaning |
|-------|-------------|-----------------|
| Dark blue / black | Low | Open space or line-of-sight path |
| Yellow | Medium | Some reflections — mid-room |
| Red / bright | High | Strong scattering — near walls or large objects |

### Pattern recognition
| Pattern | Physical cause |
|---------|----------------|
| Bright band around floor plan edge | Room walls — reflections from flat surfaces |
| Extra-bright corners | Corner = two wall reflections simultaneously |
| One corner much brighter | Transmitter location |
| Dark oval in center | Open floor — clean direct signal |
| Bright blob in middle | Large piece of furniture (desk, wardrobe, shelf) |
| Asymmetric left/right | Objects concentrated on one side |
| Gradient brightening toward one side | Transmitter is on that side |

### Tips for better data quality
- Collect with the room in its **normal furnished state** — furniture is the whole point.
- Stand **at least 50 cm away** from the ESP32 while capturing — your body absorbs WiFi.
- Hold the ESP32 at **the same height** at every position for consistent readings.
- **Keep everyone else out** of the room during collection — people create "RF shadows."
- Close windows — outdoor RF changes over time and adds noise.
- Increase `packets_per_point` to 500 for better per-cell statistics.

---

## Command Cheat Sheet

```bash
# One-time: start Ollama AI backend (keep terminal open)
ollama serve

# Verify serial stream (optional pre-check)
.venv/bin/python3 serial_monitor.py

# Full guided room mapping (ESP32 required)
.venv/bin/python3 main.py

# Generate Plotly charts from existing data (room in cm)
.venv/bin/python3 main.py --mode visualize --room 500x400x250

# Simulate a room (no hardware)
.venv/bin/python3 main.py --mode simulate --room 500x400x250

# Convert collected data to pkl then view
.venv/bin/python3 tools/esp32_to_pkl.py --input csi_data --room 5x4x2.5
.venv/bin/python3 main.py --mode load --file esp32_session.pkl

# macOS passive WiFi capture
sudo .venv/bin/python3 main.py --mode capture --interface en0 --duration 60 --room 500x400x250
```
