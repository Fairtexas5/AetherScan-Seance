"""
serial_utils.py
Auto-detects ESP32 serial port and provides robust serial reading utilities.
No API key required.
"""

import serial
import serial.tools.list_ports
import time
import sys

from pipeline.csi_parser import CSI_TAG, CSI_IQ_START


# Known USB-UART chip descriptions for ESP32
ESP32_KEYWORDS = ["CP210", "CH340", "CH341", "FTDI", "USB Serial", "ESP32", "Silicon Labs"]


def auto_detect_port() -> str | None:
    """
    Scan all available serial ports and return the most likely ESP32 port.
    Returns port string (e.g. 'COM3' or '/dev/ttyUSB0') or None if not found.
    """
    ports = serial.tools.list_ports.comports()

    if not ports:
        print("[serial_utils] No serial ports found at all.")
        return None

    print(f"[serial_utils] Found {len(ports)} port(s):")
    for p in ports:
        print(f"   {p.device:20s} | {p.description}")

    # First pass: match known ESP32/USB-UART keywords
    for p in ports:
        desc = (p.description or "") + (p.manufacturer or "")
        if any(kw.lower() in desc.lower() for kw in ESP32_KEYWORDS):
            print(f"[serial_utils] Auto-selected: {p.device} ({p.description})")
            return p.device

    # Fallback: return first available port
    fallback = ports[0].device
    print(f"[serial_utils] No ESP32 keyword match. Falling back to: {fallback}")
    return fallback


def open_serial(port: str, baud: int = 115200, timeout: float = 5.0) -> serial.Serial:
    """Open and return a configured serial connection (115200 baud = ESP-IDF Monitor default)."""
    ser = serial.Serial(port, baud, timeout=timeout)
    time.sleep(2)  # Allow ESP32 to stabilize after connect
    ser.reset_input_buffer()
    print(f"[serial_utils] Connected to {port} @ {baud} baud.")
    return ser


def validate_csi_line(line: str) -> bool:
    """
    Lightweight structural check before a full parse.
    Verifies the line starts with CSI_DATA, has at least CSI_IQ_START+2 fields,
    and that the RSSI/timestamp fields are numeric.
    """
    if not line.startswith(CSI_TAG):
        return False
    parts = line.split(",")
    if len(parts) < CSI_IQ_START + 2:   # need at least 1 IQ pair after header
        return False
    try:
        int(parts[2])   # rssi
        int(parts[3])   # n_sub
    except (ValueError, IndexError):
        return False
    return True


def read_csi_line(ser: serial.Serial) -> str | None:
    """
    Read one line from serial.
    Returns the line if it passes structural CSI_DATA validation, else None.
    Non-blocking if timeout is set on the serial object.
    """
    try:
        raw = ser.readline()
        line = raw.decode("utf-8", errors="ignore").strip()
        if validate_csi_line(line):
            return line
    except Exception as e:
        print(f"[serial_utils] Read error: {e}")
    return None


def flush_stale_data(ser: serial.Serial, flush_seconds: float = 1.0):
    """Discard any buffered serial data before starting fresh collection."""
    ser.reset_input_buffer()
    deadline = time.time() + flush_seconds
    while time.time() < deadline:
        try:
            ser.readline()
        except Exception:
            break
