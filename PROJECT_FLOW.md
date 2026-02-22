# 📡 ESP32 CSI Room Mapper — Project Flow

---

## 1. System Architecture Overview

```mermaid
graph TD
    A["📱 Phone Hotspot<br/><i>SSID: ashes</i>"]
    B["📟 ESP32<br/><i>app_main.c</i>"]
    C["💻 MacBook Pro<br/><i>USB Serial</i>"]
    D["🐍 Python Pipeline<br/><i>src/</i>"]
    E["🤖 Ollama AI<br/><i>phi3.5:latest</i>"]
    F["📊 Output Charts<br/><i>output/</i>"]

    A -- "WiFi (2.4GHz)" --> B
    B -- "CSI_DATA lines @ 115200 baud" --> C
    C -- "pyserial" --> D
    D -- "LangChain + LangGraph" --> E
    E -- "validation + anomaly reports" --> D
    D -- "Plotly HTML + PNG" --> F
```

---

## 2. Hardware Layer

```mermaid
flowchart LR
    H1["📱 Phone<br/>Hotspot ON<br/><i>SSID: ashes<br/>Pass: 987654321</i>"]
    H2["📟 ESP32<br/>app_main.c<br/><i>connects, sniffs CSI</i>"]
    H3["🔌 USB Cable<br/>to MacBook"]
    H4["🖥️ VS Code<br/>ESP_IDF: Monitor"]

    H1 -- "beacon frames<br/>every ~100ms" --> H2
    H2 -- "captures 64-subcarrier<br/>IQ CSI per packet" --> H3
    H3 --> H4
    H4 -. "close Ctrl+X<br/>before Python" .-> H3
```

### What the ESP32 emits on serial

```
CSI_DATA, {timestamp_µs}, {rssi_dBm}, {n_sub}, {extra1}, {extra2}, {I0},{Q0},{I1},{Q1},...
  Field 0      Field 1        Field 2    Field 3   Field 4   Field 5   Field 6 onwards
```

- `n_sub` = 64 subcarriers → **128 IQ values** per packet
- Parsed by `pipeline/csi_parser.py` at field index **6** (after the 6 header fields)

---

## 3. Python Project Structure

```
csi_hotspot/
├── main/
│   └── app_main.c          ← ESP32 firmware (C)
│
└── src/
    ├── main.py             ← Entry point — 5 modes
    ├── serial_monitor.py   ← Standalone ESP32 stream viewer
    ├── test_pipeline.py    ← End-to-end test (no hardware)
    ├── requirements.txt
    │
    └── pipeline/
        ├── csi_parser.py      ← Shared IQ parser (single source of truth)
        ├── serial_utils.py    ← Auto-detect port, read + validate lines
        ├── nodes.py           ← LangGraph node functions
        ├── graph.py           ← LangGraph state machine definition
        ├── validator.py       ← Statistical CSI quality check
        ├── ai_validator.py    ← LLM-based quality check (phi3.5)
        ├── ai_anomaly.py      ← LLM anomaly detection + explanation
        ├── ai_interpreter.py  ← Final LLM room report
        ├── ai_client.py       ← Ollama client config
        ├── heatmap.py         ← Matplotlib heatmap generation
        ├── csi_visualizer.py  ← Plotly 2D/3D charts
        ├── visualizer_3d.py   ← Plotly 3D voxel viewer
        ├── simulator.py       ← Synthetic CSI data (no hardware)
        └── capture_macos.py   ← WiFi packet capture via Wireshark
    │
    └── tools/
        ├── collect_positions.py  ← Guided walk-around capture
        └── esp32_to_pkl.py       ← Convert CSV → .pkl for 3D viewer
```

---

## 4. Main Pipeline Flow (LangGraph State Machine)

```mermaid
flowchart TD
    START(["▶ python main.py"])
    SETUP["node_setup\n─────────\nAuto-detect serial port\nCreate csi_data/ folder\nPrint grid info"]
    COLLECT["node_collect_point\n─────────────────\nOpen serial port\nFlush stale data\nCollect N packets\nShow pkt/s + ETA"]
    AIVAL["node_ai_validate_point\n──────────────────────\nStatistical check:\n• ≥50 packets\n• RSSI in range\n• Variance OK\nLLM confirmation via Ollama"]
    FAIL{{"✗ FAIL\n(retry < 3)"}}
    PASS{{"✓ PASS"}}
    ANOMALY["node_ai_anomaly_detection\n──────────────────────────\nCompute stats for point\nCompare to all prior points\nAsk LLM: normal or anomalous?\nPrint explanation"]
    PREVIEW["node_live_preview\n─────────────────\nRe-generate partial\nheatmap PNG preview\n(shows progress so far)"]
    ADVANCE["node_advance\n────────────\nrow++, col++\nSave CSV file\nr??_c??.csv"]
    DONE{{"All points\ncollected?"}}
    HEATMAP["node_process_heatmap\n─────────────────────\nBuild full 2D grid\nGaussian smooth\nEdge detection\nSave room_heatmap_final.png\nLaunch 3D viewer HTML"]
    INTERP["node_ai_interpret_heatmap\n──────────────────────────\nCompute grid statistics\nSummarise anomalies\nAsk LLM: interpret the room\nSave ai_report.txt"]
    END(["🏁 Done"])

    START --> SETUP --> COLLECT --> AIVAL
    AIVAL -- FAIL --> FAIL --> COLLECT
    AIVAL -- PASS --> PASS --> ANOMALY --> PREVIEW --> ADVANCE
    ADVANCE -- more points --> DONE
    DONE -- no --> COLLECT
    DONE -- yes --> HEATMAP --> INTERP --> END
```

---

## 5. CSI Parsing Pipeline

```mermaid
flowchart LR
    RAW["Raw serial line\nCSI_DATA,23503...,\n-50,64,0,0,13,-12,..."]
    VAL["serial_utils.py\nvalidate_csi_line()\n• starts with CSI_DATA?\n• ≥ 8 fields?\n• numeric RSSI?"]
    PARSE["csi_parser.py\nparse_line()\n• split fields\n• IQ = fields[6:]\n• amplitude = √(I²+Q²)"]
    OUT["ParsedResult dict\n────────────────\ntimestamp_us\nrssi\nn_sub\nextra1, extra2\niq_pairs (list)\namplitude (ndarray)"]

    RAW --> VAL
    VAL -- invalid → None --> SKIP(["skip"])
    VAL -- valid --> PARSE --> OUT
```

---

## 6. AI Layer

```mermaid
flowchart TD
    OLLAMA["🤖 Ollama Server\nlocalhost:11434\nphi3.5:latest"]

    V["ai_validator.py\n──────────────\nIs the data quality\ngood enough to keep?\nRe-ask up to 3×"]
    AN["ai_anomaly.py\n───────────────\nIs this point unusual\ncompared to the rest?\nWhat might cause it?"]
    IN["ai_interpreter.py\n──────────────────\nInterpret the full room:\n• High RF zones\n• Low RF zones\n• Anomalies found\n• 2 improvement tips"]

    OLLAMA --> V
    OLLAMA --> AN
    OLLAMA --> IN

    V -- "✓ PASS / ✗ FAIL" --> PIPE["LangGraph\npipeline"]
    AN -- "NORMAL / ANOMALOUS\n+ reason sentence" --> PIPE
    IN -- "Plain-English\nreport → ai_report.txt" --> PIPE
```

---

## 7. Five Run Modes

```mermaid
flowchart LR
    MAIN["python main.py"]

    M1["(no flag)\n────────\nAI Pipeline Mode\nFull LangGraph run\nESP32 required"]
    M2["--mode simulate\n───────────────\nSynthetic CSI data\nNo hardware needed\nGood first test"]
    M3["--mode capture\n──────────────\nmacOS WiFi sniff\nvia Wireshark/tshark\nsudo required"]
    M4["--mode load\n───────────\nLoad saved .pkl\nfile and view\nin 3D viewer"]
    M5["--mode visualize\n─────────────────\nBuild 3 Plotly charts\nfrom csi_data/ CSVs\nNo hardware needed"]

    MAIN --> M1
    MAIN --> M2
    MAIN --> M3
    MAIN --> M4
    MAIN --> M5
```

### CLI Examples

```bash
# Full pipeline (real ESP32):
python main.py

# Simulate a 500×400×250cm room:
python main.py --mode simulate --room 500x400x250

# Generate 3 Plotly charts from collected data:
python main.py --mode visualize --room 500x400x250 --data csi_data

# Capture via Wireshark (macOS):
sudo python main.py --mode capture --interface en0 --duration 60 --room 500x400x250

# Load + view a saved session:
python main.py --mode load --file esp32_session.pkl
```

---

## 8. Output Files

```mermaid
flowchart LR
    PIPE["Pipeline\nCompletes"]

    P1["csi_data/\nr00_c00.csv ... ← raw packets per cell"]
    P2["output/room_heatmap_final.png\n← Matplotlib 2D heatmap (cm axes)"]
    P3["output/room_3d.html\n← Plotly 3D voxel viewer"]
    P4["output/csi_2d_heatmap.html\n← Interactive 2D Plotly heatmap"]
    P5["output/csi_3d_surface.html\n← 3D surface heatmap"]
    P6["output/csi_3d_room.html\n← 3D room structure + measurement points"]
    P7["output/ai_report.txt\n← Plain-English AI interpretation"]

    PIPE --> P1
    PIPE --> P2
    PIPE --> P3
    PIPE --> P4
    PIPE --> P5
    PIPE --> P6
    PIPE --> P7
```

---

## 9. Room-to-Grid Coordinate System

```
Phone hotspot (TX)
  ★
  │
  │  ← depth (room_height_m)
  │
  ▼
Col→  0        1        2       ...    (N-1)
Row 0 [ 0,0 ] [ 0,1 ] [ 0,2 ]  ...   saves as r00_c00.csv
Row 1 [ 1,0 ] [ 1,1 ] [ 1,2 ]  ...
Row 2 [ 2,0 ] [ 2,1 ] [ 2,2 ]  ...
 ↓
(M-1)
```

**Using floor tiles:** count grout lines (each 60cm tile = 1 grid step).
**Using steps:** ~75cm per stride = use as 1 grid step at `grid_cols/rows` matching.

---

## 10. Quick Start Checklist

```
[ ] Phone hotspot "ashes" is ON
[ ] ESP32 plugged into MacBook via USB
[ ] VS Code → ESP_IDF: Monitor → see CSI_DATA lines → Ctrl+X to close
[ ] Terminal 1: ollama serve  (keep open)
[ ] Terminal 2: cd src && .venv/bin/python3 serial_monitor.py  (check stream)
[ ] Edit src/main.py CONFIG: grid_rows, grid_cols, room_width_m, room_height_m
[ ] Terminal 2: .venv/bin/python3 main.py  (start collection)
[ ] Walk to each grid point, press Enter, wait for AI validation
[ ] After all points: charts auto-open in browser
[ ] .venv/bin/python3 main.py --mode visualize --room 500x400x250
```
