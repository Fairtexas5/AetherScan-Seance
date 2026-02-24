# AetherScan-Seance

> **WiFi CSI-based indoor room mapper using ESP32 + local AI (Ollama)**
> No cameras. No LiDAR. Just WiFi signal fingerprints.

---

## What It Does

AetherScan-Seance uses a single ESP32 to capture **Channel State Information (CSI)** — the complex IQ samples hidden inside every WiFi packet. By measuring signal variance across a floor grid, it builds a **2D + 3D RF map** of a room, detecting walls, furniture, and open spaces.

```
Phone Hotspot ──WiFi──► ESP32 ──USB──► Python Pipeline ──► AI + Heatmaps
```

---

## Features

- **Real-time CSI capture** from ESP32 at 115200 baud
- **Local AI validation & anomaly detection** via Ollama (`phi3.5:latest`) — no API key
- **Live heatmap preview** after every grid point
- **Three Plotly interactive charts** — 2D heatmap, 3D surface, 3D room structure
- **LangGraph state machine** — structured collection → validate → anomaly → repeat

---

## Hardware Required

| Component | Details |
|-----------|---------|
| ESP32 dev board | Any with USB (NodeMCU-32S, DOIT, etc.) |
| USB cable | Data-capable (not charge-only) |
| MacBook / Windows | macOS, Python 3.11+ |
| Phone hotspot | Fixed SSID & password (configured in firmware) |

---

## Project Structure

```
csi_hotspot/
├── main/
│   └── app_main.c          ← ESP32 firmware (C, ESP-IDF)
│
└── src/
    ├── main.py             ← Entry point — 5 run modes
    ├── serial_monitor.py   ← Standalone stream viewer
    ├── chair_demo.py       ← Demo charts from photo grid
    ├── test_pipeline.py    ← End-to-end test (no hardware)
    ├── requirements.txt
    ├── ROOM_MAPPING_GUIDE.md
    │
    ├── pipeline/
    │   ├── csi_parser.py      ← Shared IQ parser (index 6 offset)
    │   ├── serial_utils.py    ← Auto-detect port, validate lines
    │   ├── graph.py           ← LangGraph state machine
    │   ├── nodes.py           ← Pipeline node functions
    │   ├── validator.py       ← Statistical quality check
    │   ├── ai_validator.py    ← LLM quality validation
    │   ├── ai_anomaly.py      ← LLM anomaly detection
    │   ├── ai_interpreter.py  ← LLM final room report
    │   ├── ai_client.py       ← Ollama client config
    │   ├── heatmap.py         ← Matplotlib heatmap
    │   ├── csi_visualizer.py  ← Plotly 2D/3D charts
    │   ├── visualizer_3d.py   ← Plotly 3D voxel viewer
    │   ├── simulator.py       ← Synthetic CSI (no hardware)
    │   └── capture_macos.py   ← WiFi sniff via Wireshark/esp32
    │
    └── tools/
        ├── collect_positions.py  ← Guided walk-around capture
        └── esp32_to_pkl.py       ← CSV → .pkl converter
```

---

## Quick Start

### 1. Install dependencies

```bash
cd src
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Flash the ESP32

Open the project in VS Code with ESP-IDF extension, build and flash `main/app_main.c`.
Update the WiFi credentials in `app_main.c` to match your phone hotspot.

### 3. Start Ollama AI

```bash
ollama serve
ollama pull phi3.5:latest
```

### 4. Run the full pipeline

```bash
cd src
.venv/bin/python3 main.py
```

Follow the on-screen prompts — carry the ESP32 to each grid position, press Enter.

### 5. Generate visualizations

```bash
.venv/bin/python3 main.py --mode visualize --room 500x400x250
```

---

## Run Modes

| Command | What it does |
|---------|-------------|
| `python main.py` | Full AI pipeline (ESP32 required) |
| `python main.py --mode simulate --room 500x400x250` | Synthetic data, no hardware |
| `python main.py --mode visualize --room 500x400x250` | Build Plotly charts from collected CSVs |
| `python main.py --mode capture --interface en0` | WiFi sniff via esp32 (macOS) |
| `python main.py --mode load --file session.pkl` | Load + view a saved session |

---

## Output Files

| File | Description |
|------|-------------|
| `output/room_heatmap_final.png` | Matplotlib 2D heatmap (cm axes) |
| `output/csi_2d_heatmap.html` | Plotly interactive 2D heatmap |
| `output/csi_3d_surface.html` | Plotly 3D surface chart |
| `output/csi_3d_room.html` | 3D room + measurement points |
| `output/chair_3d_room.html` | 3D room demo from `chair_demo.py` (photo grid) |
| `output/ai_report.txt` | AI plain-English interpretation |
| `csi_data/r??_c??.csv` | Raw CSI packets per grid position |

---

## CSI Data Format

Each line emitted by the ESP32 firmware:

```
CSI_DATA, {timestamp_µs}, {rssi_dBm}, {n_sub}, {extra1}, {extra2}, {I0},{Q0},{I1},{Q1},...
```

- `n_sub` = 64 subcarriers → 128 IQ values per packet
- IQ data starts at **field index 6** (parsed by `pipeline/csi_parser.py`)
- Amplitude = √(I² + Q²) per subcarrier

---

## Using Floor Tiles as a Grid

No need to mark the floor — count the grout lines:

| Tile size | Grid step |
|-----------|-----------|
| 60 × 60 cm | 1 tile per cell |
| 30 × 30 cm | 2 tiles per cell |

Configure `grid_rows`, `grid_cols`, `room_width_m`, `room_height_m` in `src/main.py` `CONFIG`.

---

## Testing (No Hardware)

```bash
cd src
MOCK_AI=1 .venv/bin/python3 test_pipeline.py
```

---

## Dependencies

See `src/requirements.txt`. Key packages:

```
langgraph · langchain-ollama · langchain-core
pyserial · numpy · scipy · matplotlib
pandas · plotly · pyshark
```

---

## See Also

- [`PROJECT_FLOW.md`](PROJECT_FLOW.md) — Full system architecture diagrams
- [`src/ROOM_MAPPING_GUIDE.md`](src/ROOM_MAPPING_GUIDE.md) — Step-by-step room mapping guide

---

## License

Academic / research use. Hardware design and firmware are custom-built for this project.
