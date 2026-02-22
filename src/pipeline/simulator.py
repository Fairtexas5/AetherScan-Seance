"""
simulator.py
─────────────────────────────────────────────────────────
Generates physically plausible synthetic CSI / RSSI data
for the WiFi CSI 3D Room Mapper.

No hardware required — uses radio propagation physics:
  • Free-space path loss model (FSPL)
  • Wall reflections: boundary voxels get +6 dB gain
  • Random furniture blobs: elevated RF scatter
  • Returns a 3-D numpy voxel array + a flat list of
    synthetic CSI_DATA strings compatible with the rest
    of the pipeline.

Usage:
    from pipeline.simulator import simulate_room
    voxel_grid, frames, shape = simulate_room(
        room_w=5.0, room_d=4.0, room_h=2.5,
        grid_res=0.25,   # metres per voxel
        tx_pos=(0.0, 0.0, 1.0)  # transmitter corner
    )
    # Returns: (voxel_grid ndarray, CSI_DATA frames list, (nW, nD, nH) shape tuple)
─────────────────────────────────────────────────────────
"""

import time
import random
import numpy as np

# ── Number of synthetic CSI subcarriers (matches ESP32 output) ──
N_SUBCARRIERS = 52


def _free_space_path_loss(d_m: float, freq_ghz: float = 2.4) -> float:
    """FSPL in dB: 20·log10(d) + 20·log10(f) + 92.45 for GHz/km."""
    d = max(d_m, 0.01)
    return 20.0 * np.log10(d) + 20.0 * np.log10(freq_ghz * 1e9) - 147.55


def _rssi_from_fspl(fspl_db: float, tx_power_dbm: float = 20.0) -> float:
    """Convert FSPL → received RSSI in dBm."""
    return tx_power_dbm - fspl_db


def simulate_room(
    room_w: float = 5.0,
    room_d: float = 4.0,
    room_h: float = 2.5,
    grid_res: float = 0.25,
    tx_pos: tuple = (0.0, 0.0, 1.0),
    n_furniture: int = 3,
    packets_per_voxel: int = 4,
    verbose: bool = True,
) -> tuple:
    """
    Generate a synthetic 3-D RF voxel map and matching CSI_DATA strings.

    Parameters
    ----------
    room_w, room_d, room_h : room dimensions in metres
    grid_res               : voxel side length in metres (smaller = finer)
    tx_pos                 : (x, y, z) of transmitter in metres
    n_furniture            : number of random furniture blobs to add
    packets_per_voxel      : number of synthetic CSI frames generated per voxel
    verbose                : print progress lines

    Returns
    -------
    voxel_grid : np.ndarray shape (nW, nD, nH), values = normalised RF intensity [0,1]
    frames     : list[str] of CSI_DATA lines (for compatibility with pipeline parsers)
    """
    nW = max(1, int(round(room_w / grid_res)))
    nD = max(1, int(round(room_d / grid_res)))
    nH = max(1, int(round(room_h / grid_res)))

    voxels = np.zeros((nW, nD, nH), dtype=float)

    # ── Transmitter voxel position ──────────────────────────────────
    tx_ix = min(int(tx_pos[0] / grid_res), nW - 1)
    tx_iy = min(int(tx_pos[1] / grid_res), nD - 1)
    tx_iz = min(int(tx_pos[2] / grid_res), nH - 1)

    # ── Random furniture blobs ──────────────────────────────────────
    furniture_centres = []
    for _ in range(n_furniture):
        fx = random.uniform(0.3 * nW, 0.7 * nW)
        fy = random.uniform(0.3 * nD, 0.7 * nD)
        fz = random.uniform(0.0,      0.5 * nH)
        fr = random.uniform(0.5 / grid_res, 1.2 / grid_res)  # blob radius in voxels
        furniture_centres.append((fx, fy, fz, fr))

    total_voxels = nW * nD * nH
    frames: list[str] = []
    frame_count = 0
    progress_step = max(1, total_voxels // 10)

    for ix in range(nW):
        for iy in range(nD):
            for iz in range(nH):
                # Real-world position (metres, voxel centre)
                x = (ix + 0.5) * grid_res
                y = (iy + 0.5) * grid_res
                z = (iz + 0.5) * grid_res

                # Distance from transmitter
                dx = x - tx_pos[0]
                dy = y - tx_pos[1]
                dz = z - tx_pos[2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                # Base RSSI from free-space path loss
                rssi = _rssi_from_fspl(_free_space_path_loss(dist))

                # Wall bonus: voxels touching a room boundary reflect strongly
                wall_bonus = 0.0
                if ix == 0 or ix == nW - 1:
                    wall_bonus += 6.0
                if iy == 0 or iy == nD - 1:
                    wall_bonus += 6.0
                if iz == 0 or iz == nH - 1:
                    wall_bonus += 3.0

                # Furniture bonus
                furn_bonus = 0.0
                for fx, fy, fz, fr in furniture_centres:
                    fd = np.sqrt((ix - fx)**2 + (iy - fy)**2 + (iz - fz)**2)
                    if fd < fr:
                        furn_bonus += 8.0 * (1.0 - fd / fr)

                # Effective signal power (dBm → linear for the voxel value)
                eff_rssi = rssi + wall_bonus + furn_bonus + random.gauss(0, 1.5)
                voxels[ix, iy, iz] = eff_rssi

                # ── Synthesise CSI_DATA packets for this voxel ──────
                for _ in range(packets_per_voxel):
                    ts = int(time.time() * 1e6) + frame_count
                    rssi_int = int(np.clip(eff_rssi, -95, -20))
                    iq_vals = []
                    base_amp = max(1, int(-eff_rssi / 2))  # amplitude ∝ signal loss
                    for _ in range(N_SUBCARRIERS):
                        i_val = int(random.gauss(base_amp, base_amp * 0.3 + 1))
                        q_val = int(random.gauss(base_amp, base_amp * 0.3 + 1))
                        iq_vals.extend([i_val, q_val])
                    # Format matches real firmware: CSI_DATA,ts,rssi,n_sub,extra1,extra2,IQ...
                    # extra1/extra2 are metadata bytes (0,0 for simulated data)
                    line = (
                        f"CSI_DATA,{ts},{rssi_int},{N_SUBCARRIERS},0,0,"
                        + ",".join(map(str, iq_vals))
                    )
                    frames.append(line)
                    frame_count += 1

                voxel_idx = ix * nD * nH + iy * nH + iz
                if verbose and voxel_idx % progress_step == 0:
                    pct = 100 * voxel_idx // total_voxels
                    print(f"  Generated {frame_count}/{total_voxels * packets_per_voxel}"
                          f" frames ({pct}%)...")

    # ── Normalise voxel grid to [0, 1] ─────────────────────────────
    # Shift so higher RSSI (less negative) → higher value
    vmin, vmax = voxels.min(), voxels.max()
    if vmax > vmin:
        voxels = (voxels - vmin) / (vmax - vmin)
    else:
        voxels = np.zeros_like(voxels)

    if verbose:
        print(f"  Generated {frame_count}/{total_voxels * packets_per_voxel} frames.  Done.")
        print(f"  Voxel grid shape: {voxels.shape}  "
              f"(W={nW}, D={nD}, H={nH})")

    return voxels, frames, (nW, nD, nH)
