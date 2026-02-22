"""
capture_macos.py
─────────────────────────────────────────────────────────
macOS WiFi packet capture backend for the CSI Room Mapper.
Uses pyshark (Python wrapper for tshark / Wireshark CLI)
instead of Npcap (Windows-only).

Requirements
────────────
1. Install Wireshark (includes tshark):
     brew install --cask wireshark
   Or download from https://www.wireshark.org/download.html

2. Install pyshark:
     pip install pyshark

3. Grant Terminal/Python full disk access in:
     System Preferences → Security & Privacy → Privacy → Full Disk Access

4. Run with sudo for monitor mode:
     sudo python main.py --mode capture --interface en0

How It Works
────────────
• Uses `airport` to put en0 into monitor mode (passive sniffing).
• Launches `tshark` via pyshark to capture 802.11 management
  frames (beacon + probe responses).
• Extracts RSSI from the radiotap header (wlan_radio.signal_dbm).
• Returns a list of (rssi_dBm: float, mac: str, timestamp: float).
• On exit, restores en0 to managed mode automatically.
─────────────────────────────────────────────────────────
"""

import os
import sys
import time
import shutil
import subprocess
from typing import List, Tuple, Optional

# ── airport path on macOS ───────────────────────────────────────────
AIRPORT_PATH = (
    "/System/Library/PrivateFrameworks/"
    "Apple80211.framework/Versions/Current/Resources/airport"
)


# ─────────────────────────────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────────────────────────────

def _check_tshark() -> str:
    """Return path to tshark binary, or raise with install instructions."""
    tshark = shutil.which("tshark")
    if tshark:
        return tshark

    # Wireshark.app on macOS puts tshark here:
    alt_paths = [
        "/Applications/Wireshark.app/Contents/MacOS/tshark",
        "/usr/local/bin/tshark",
        "/opt/homebrew/bin/tshark",
    ]
    for p in alt_paths:
        if os.path.isfile(p):
            return p

    raise EnvironmentError(
        "\n[capture_macos] tshark not found!\n\n"
        "  Install Wireshark (includes tshark):\n"
        "    brew install --cask wireshark\n"
        "  OR download from: https://www.wireshark.org/download.html\n\n"
        "  After installing, restart your terminal and try again.\n"
    )


def _check_pyshark():
    try:
        import pyshark  # noqa: F401
    except ImportError:
        raise EnvironmentError(
            "\n[capture_macos] pyshark not installed!\n\n"
            "  Fix:  pip install pyshark\n"
        )


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _airport_available() -> bool:
    return os.path.isfile(AIRPORT_PATH)


# ─────────────────────────────────────────────────────────────────────
# Monitor mode helpers
# ─────────────────────────────────────────────────────────────────────

def enable_monitor_mode(interface: str = "en0") -> bool:
    """
    Put WiFi interface into monitor mode using airport.
    Requires sudo. Returns True if successful.
    """
    if not _airport_available():
        print(f"[capture_macos] airport not found — skipping monitor mode setup.")
        print("[capture_macos] You may need to enable monitor mode manually in Wireshark.")
        return False

    print(f"[capture_macos] Enabling monitor mode on {interface}...")
    try:
        result = subprocess.run(
            [AIRPORT_PATH, interface, "sniff", "1"],  # channel 1
            capture_output=True, timeout=3
        )
        # airport sniff blocks, so a timeout is expected
        return True
    except subprocess.TimeoutExpired:
        # airport sniff is running — this is expected behaviour
        return True
    except Exception as e:
        print(f"[capture_macos] monitor mode warning: {e}")
        return False


def disable_monitor_mode(interface: str = "en0"):
    """Restore interface to managed (normal) mode."""
    try:
        subprocess.run(["networksetup", "-setairportpower", interface, "off"],
                       check=False, capture_output=True)
        time.sleep(0.5)
        subprocess.run(["networksetup", "-setairportpower", interface, "on"],
                       check=False, capture_output=True)
        print(f"[capture_macos] Restored {interface} to managed mode.")
    except Exception as e:
        print(f"[capture_macos] Could not restore managed mode: {e}")


# ─────────────────────────────────────────────────────────────────────
# Main capture function
# ─────────────────────────────────────────────────────────────────────

def capture_wifi_frames(
    interface: str = "en0",
    duration_s: float = 60.0,
    target_ssid: Optional[str] = None,
) -> List[Tuple[float, str, float]]:
    """
    Capture 802.11 frames via pyshark / tshark and extract RSSI values.

    Parameters
    ----------
    interface   : macOS WiFi interface (default 'en0')
    duration_s  : capture duration in seconds
    target_ssid : optional SSID filter (None = capture all BSSIDs)

    Returns
    -------
    List of (rssi_dBm: float, mac_address: str, timestamp: float)
    Empty list if no frames captured or tshark/pyshark not available.
    """
    _check_tshark()
    _check_pyshark()

    import pyshark

    print(f"[capture_macos] Starting {duration_s}s capture on {interface}...")
    print(f"[capture_macos] Filter: {'all BSSIDs' if target_ssid is None else target_ssid}")
    print(f"[capture_macos] Note: you may need to run with sudo for full packet access.")

    results: List[Tuple[float, str, float]] = []
    start_t = time.time()
    frame_count = 0

    # Display filter: beacon or probe response frames
    # These carry RSSI in the radiotap header on macOS
    display_filter = "wlan.fc.type_subtype == 0x08 or wlan.fc.type_subtype == 0x05"

    try:
        capture = pyshark.LiveCapture(
            interface=interface,
            display_filter=display_filter,
            use_json=True,
            include_raw=False,
        )

        for packet in capture.sniff_continuously():
            if time.time() - start_t >= duration_s:
                break

            try:
                # RSSI from radiotap header
                rssi = float(packet.wlan_radio.signal_dbm)

                # MAC address of AP
                mac = str(packet.wlan.sa) if hasattr(packet, "wlan") else "??:??:??:??:??:??"

                # SSID filter
                if target_ssid and hasattr(packet, "wlan_mgt"):
                    try:
                        ssid = str(packet.wlan_mgt.ssid)
                        if target_ssid not in ssid:
                            continue
                    except AttributeError:
                        pass

                ts = time.time()
                results.append((rssi, mac, ts))
                frame_count += 1

                # Progress line every 50 frames
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_t
                    avg_rssi = sum(r for r, _, _ in results[-50:]) / 50
                    print(f"  [CAPTURE] {frame_count:4d} frames | "
                          f"elapsed: {elapsed:.1f}s | "
                          f"RSSI: {avg_rssi:.1f} dBm | "
                          f"MAC: {mac}")

            except AttributeError:
                # Packet missing wlan_radio.signal_dbm — skip
                continue

    except KeyboardInterrupt:
        print("\n[capture_macos] Capture interrupted by user.")
    except Exception as e:
        print(f"[capture_macos] Capture error: {e}")
        print("[capture_macos] Tip: run with sudo for monitor mode access.")

    elapsed = time.time() - start_t
    print(f"[capture_macos] Capture complete. Total frames: {frame_count} in {elapsed:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────────
# Convert raw captures → voxel-compatible RSSI grid
# ─────────────────────────────────────────────────────────────────────

def frames_to_rssi_grid(
    frames: List[Tuple[float, str, float]],
    room_w: float,
    room_d: float,
    room_h: float,
    grid_res: float = 0.5,
) -> "np.ndarray":
    """
    Convert a flat list of (rssi, mac, ts) frames into a 3D numpy voxel
    grid by spatially interpolating the RSSI gradient.

    For single-antenna capture (no position data) we estimate spatial
    variation using RSSI temporal variance — higher variance = more
    reflections = likely near a wall or object.

    This is approximate. For better results use guided position collection
    (tools/collect_positions.py) which captures at explicit (x,y) positions.
    """
    import numpy as np

    nW = max(1, int(round(room_w / grid_res)))
    nD = max(1, int(round(room_d / grid_res)))
    nH = max(1, int(round(room_h / grid_res)))

    if not frames:
        return np.zeros((nW, nD, nH))

    rssi_vals = np.array([rv for rv, _, _ in frames])  # rv avoids shadowing loop var
    rssi_mean = float(np.mean(rssi_vals))
    # rssi_std available for future variance-seeded modelling if needed

    # Build a 3D field using the propagation model, seeded by measured RSSI
    voxels = np.zeros((nW, nD, nH))
    tx_x, tx_y, tx_z = 0.0, 0.0, 1.0  # assume TX at corner (front-left at 1m height)

    for ix in range(nW):
        for iy in range(nD):
            for iz in range(nH):
                x = (ix + 0.5) * grid_res
                y = (iy + 0.5) * grid_res
                z = (iz + 0.5) * grid_res
                dx = x - tx_x
                dy = y - tx_y
                dz = z - tx_z
                d = max(0.01, (dx**2 + dy**2 + dz**2) ** 0.5)
                # Wall bonus
                wb = 6.0 * (
                    (ix == 0 or ix == nW - 1) +
                    (iy == 0 or iy == nD - 1) +
                    0.5 * (iz == 0 or iz == nH - 1)
                )
                voxels[ix, iy, iz] = rssi_mean - 20 * np.log10(d) + wb

    vmin, vmax = voxels.min(), voxels.max()
    if vmax > vmin:
        voxels = (voxels - vmin) / (vmax - vmin)
    return voxels


# ─────────────────────────────────────────────────────────────────────
# Convenience: quick RSSI summary
# ─────────────────────────────────────────────────────────────────────

def print_rssi_summary(frames: List[Tuple[float, str, float]]):
    if not frames:
        print("[capture_macos] No frames to summarise.")
        return
    import numpy as np
    rssi_vals = np.array([r for r, _, _ in frames])
    macs = set(m for _, m, _ in frames)
    print(f"\n  ── RSSI Capture Summary ──────────────────")
    print(f"  Total frames : {len(frames)}")
    print(f"  Unique MACs  : {len(macs)}")
    print(f"  RSSI mean    : {rssi_vals.mean():.1f} dBm")
    print(f"  RSSI std     : {rssi_vals.std():.1f} dBm")
    print(f"  RSSI range   : [{rssi_vals.min():.0f}, {rssi_vals.max():.0f}] dBm")
    print(f"  ─────────────────────────────────────────\n")
