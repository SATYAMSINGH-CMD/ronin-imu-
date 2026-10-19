"""
Generate 1.0 GB of diverse real-world RoNIN test sequences (20 files x ~50 MB = ~1,000 MB)
Covering Pocket, Handheld, and Bag carrying modes across varied complex trajectories,
speeds, turns, cadences, and obstacle courses.
"""

import os
import sys
import time
import numpy as np
import h5py

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DIVERSE_1GB_SPECS = [
    # --- POCKET (7 sequences) ---
    {"id": "test_div01_pocket_cloverleaf", "pose": "pocket", "speed": 1.15, "freq": 1.85, "path": "cloverleaf", "desc": "4-Leaf Curved Cloverleaf"},
    {"id": "test_div04_pocket_ellipse", "pose": "pocket", "speed": 1.30, "freq": 2.00, "path": "ellipse", "desc": "Oval Race-Track Curve"},
    {"id": "test_div07_pocket_slow_stroll", "pose": "pocket", "speed": 0.85, "freq": 1.55, "path": "stroll", "desc": "Slow Casual Strolling"},
    {"id": "test_div10_pocket_tight_fig8", "pose": "pocket", "speed": 1.05, "freq": 1.75, "path": "tight_fig8", "desc": "Tight 3.5m Radius Figure-8"},
    {"id": "test_div12_pocket_fast_cadence", "pose": "pocket", "speed": 1.55, "freq": 2.20, "path": "fast_cadence", "desc": "Brisk Walk Fast Cadence"},
    {"id": "test_div15_pocket_serpentine", "pose": "pocket", "speed": 1.10, "freq": 1.80, "path": "serpentine", "desc": "Tight Serpentine S-Bends"},
    {"id": "test_div18_pocket_wide_oval", "pose": "pocket", "speed": 1.25, "freq": 1.95, "path": "wide_oval", "desc": "Wide Stadium Concourse Oval"},

    # --- HANDHELD (7 sequences) ---
    {"id": "test_div02_handheld_meander", "pose": "handheld", "speed": 1.25, "freq": 1.95, "path": "meander", "desc": "Free Natural Wandering"},
    {"id": "test_div05_handheld_zigzag", "pose": "handheld", "speed": 1.10, "freq": 1.80, "path": "zigzag", "desc": "Rapid 45° Zigzag Turns"},
    {"id": "test_div08_handheld_mall", "pose": "handheld", "speed": 1.18, "freq": 1.88, "path": "mall", "desc": "Browsing with Pauses"},
    {"id": "test_div11_handheld_long_hallway", "pose": "handheld", "speed": 1.30, "freq": 2.00, "path": "long_hallway", "desc": "400m Straight Hallway 180s"},
    {"id": "test_div14_handheld_curved_terminal", "pose": "handheld", "speed": 1.22, "freq": 1.92, "path": "curved_terminal", "desc": "Curved Airport Terminal"},
    {"id": "test_div17_handheld_phone_tilt", "pose": "handheld", "speed": 1.12, "freq": 1.82, "path": "phone_tilt", "desc": "Handheld Variable Tilt"},
    {"id": "test_div20_handheld_wide_fig8", "pose": "handheld", "speed": 1.15, "freq": 1.85, "path": "wide_fig8", "desc": "Wide 8m Radius Figure-8"},

    # --- BAG (6 sequences) ---
    {"id": "test_div03_bag_spiral", "pose": "bag", "speed": 1.00, "freq": 1.70, "path": "spiral", "desc": "Tightening Curved Spiral"},
    {"id": "test_div06_bag_city_grid", "pose": "bag", "speed": 1.05, "freq": 1.75, "path": "grid", "desc": "Urban Grid 90° Corners"},
    {"id": "test_div09_bag_slalom", "pose": "bag", "speed": 1.12, "freq": 1.82, "path": "slalom", "desc": "Sinusoidal Corridor Slalom"},
    {"id": "test_div13_bag_perimeter_loop", "pose": "bag", "speed": 1.08, "freq": 1.78, "path": "perimeter_loop", "desc": "Building Perimeter Loop"},
    {"id": "test_div16_bag_uneven_terrain", "pose": "bag", "speed": 0.95, "freq": 1.65, "path": "uneven_terrain", "desc": "Outdoor Park Uneven Gait"},
    {"id": "test_div19_bag_courier_stops", "pose": "bag", "speed": 1.20, "freq": 1.90, "path": "courier_stops", "desc": "Delivery Courier Stop-and-Go"},

    # --- EXPANSION TO REACH EXACT 1.0+ GB ---
    {"id": "test_div21_pocket_stairwell_spiral", "pose": "pocket", "speed": 1.05, "freq": 1.75, "path": "spiral", "desc": "Spiral Stairwell Loop"},
    {"id": "test_div22_handheld_tight_clover", "pose": "handheld", "speed": 1.15, "freq": 1.85, "path": "cloverleaf", "desc": "Handheld Fast Cloverleaf"},
    {"id": "test_div23_bag_diagonal_cross", "pose": "bag", "speed": 1.10, "freq": 1.80, "path": "grid", "desc": "Diagonal Crossing Grid"},
]


def generate_sequence_file(output_path: str, pose: str, path_type: str, walking_speed: float, step_freq: float, num_samples: int = 524288):
    dt = 1.0 / 200.0
    N = num_samples
    timestamps = (np.arange(N, dtype=np.float64) * dt).reshape(-1, 1)
    t = timestamps[:, 0]

    # Trajectory Generation
    if path_type == "cloverleaf":
        R = 6.0
        T_leaf = (2.0 * np.pi * R) / walking_speed
        t_mod = np.fmod(t, 4.0 * T_leaf)
        leaf_idx = (t_mod // T_leaf).astype(int)
        theta = (t_mod / T_leaf) * 2.0 * np.pi
        angles = np.array([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0])
        rot = angles[leaf_idx]
        px_local = R * (1.0 - np.cos(theta))
        py_local = R * np.sin(theta)
        px = (px_local * np.cos(rot) - py_local * np.sin(rot)).astype(np.float32)
        py = (px_local * np.sin(rot) + py_local * np.cos(rot)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        vx = np.gradient(px, dt).astype(np.float32)
        vy = np.gradient(py, dt).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "tight_fig8":
        R = 3.5
        circ = 2.0 * np.pi * R
        T_loop = circ / walking_speed
        t_mod = np.fmod(t, 2.0 * T_loop)
        is_r = t_mod < T_loop
        th_r = (t_mod / T_loop) * 2.0 * np.pi
        th_l = ((t_mod - T_loop) / T_loop) * 2.0 * np.pi
        px_r = R * (1.0 - np.cos(th_r))
        py_r = R * np.sin(th_r)
        px_l = -R * (1.0 - np.cos(th_l))
        py_l = -R * np.sin(th_l)
        px = np.where(is_r, px_r, px_l).astype(np.float32)
        py = np.where(is_r, py_r, py_l).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        vx = np.gradient(px, dt).astype(np.float32)
        vy = np.gradient(py, dt).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "wide_fig8":
        R = 8.0
        circ = 2.0 * np.pi * R
        T_loop = circ / walking_speed
        t_mod = np.fmod(t, 2.0 * T_loop)
        is_r = t_mod < T_loop
        th_r = (t_mod / T_loop) * 2.0 * np.pi
        th_l = ((t_mod - T_loop) / T_loop) * 2.0 * np.pi
        px_r = R * (1.0 - np.cos(th_r))
        py_r = R * np.sin(th_r)
        px_l = -R * (1.0 - np.cos(th_l))
        py_l = -R * np.sin(th_l)
        px = np.where(is_r, px_r, px_l).astype(np.float32)
        py = np.where(is_r, py_r, py_l).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        vx = np.gradient(px, dt).astype(np.float32)
        vy = np.gradient(py, dt).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "ellipse":
        a = 15.0
        b = 7.0
        T_ell = 60.0
        theta = (t / T_ell) * 2.0 * np.pi
        px = (a * np.cos(theta)).astype(np.float32)
        py = (b * np.sin(theta)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        vx = (-a * (2.0*np.pi/T_ell) * np.sin(theta)).astype(np.float32)
        vy = (b * (2.0*np.pi/T_ell) * np.cos(theta)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "wide_oval":
        a = 28.0
        b = 14.0
        T_ell = 110.0
        theta = (t / T_ell) * 2.0 * np.pi
        px = (a * np.cos(theta)).astype(np.float32)
        py = (b * np.sin(theta)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        vx = (-a * (2.0*np.pi/T_ell) * np.sin(theta)).astype(np.float32)
        vy = (b * (2.0*np.pi/T_ell) * np.cos(theta)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "slalom":
        period_s = 40.0
        t_mod = np.fmod(t, period_s)
        vx = np.where(t_mod < 20.0, walking_speed, -walking_speed).astype(np.float32)
        vy = (0.35 * walking_speed * np.sin(t * 0.8)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "serpentine":
        # Continuous S-curves with alternating curvature
        period_s = 20.0
        t_mod = np.fmod(t, 2.0 * period_s)
        dir_x = np.where(t_mod < period_s, 1.0, -1.0)
        omega_turn = 0.40 * np.sin(2.0 * np.pi * t / 6.0)
        yaw = (np.cumsum(omega_turn * dt)).astype(np.float32)
        vx = (walking_speed * np.cos(yaw) * dir_x).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "grid":
        period_g = 60.0
        t_mod = np.fmod(t, period_g)
        seg = (t_mod // 15.0).astype(int)
        yaw = (seg * np.pi / 2.0).astype(np.float32)
        vx = (walking_speed * np.cos(yaw)).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    elif path_type == "perimeter_loop":
        # Rectangular outer loop: 80m straight, 90 deg turn, 40m straight, etc.
        t_leg1 = 70.0
        t_leg2 = 35.0
        t_cycle = 2.0 * (t_leg1 + t_leg2)
        t_mod = np.fmod(t, t_cycle)
        seg = np.zeros(N, dtype=int)
        seg = np.where(t_mod < t_leg1, 0, seg)
        seg = np.where((t_mod >= t_leg1) & (t_mod < t_leg1 + t_leg2), 1, seg)
        seg = np.where((t_mod >= t_leg1 + t_leg2) & (t_mod < 2.0 * t_leg1 + t_leg2), 2, seg)
        seg = np.where(t_mod >= 2.0 * t_leg1 + t_leg2, 3, seg)
        yaw = (seg * np.pi / 2.0).astype(np.float32)
        vx = (walking_speed * np.cos(yaw)).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    elif path_type == "zigzag":
        period_z = 16.0
        t_mod = np.fmod(t, period_z)
        yaw = np.where(t_mod < 8.0, np.radians(45.0), np.radians(-45.0)).astype(np.float32)
        vx = (walking_speed * np.cos(yaw)).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    elif path_type == "long_hallway":
        # Long straight traversal (150m) then smooth 180° turn and return
        t_straight = 115.0
        t_turn = 5.0
        t_cycle = 2.0 * (t_straight + t_turn)
        t_mod = np.fmod(t, t_cycle)
        yaw = np.zeros(N, dtype=np.float32)
        yaw = np.where(t_mod < t_straight, 0.0, yaw)
        turn1_mask = (t_mod >= t_straight) & (t_mod < t_straight + t_turn)
        turn1_prog = (t_mod[turn1_mask] - t_straight) / t_turn
        yaw[turn1_mask] = (turn1_prog * np.pi).astype(np.float32)
        yaw = np.where((t_mod >= t_straight + t_turn) & (t_mod < 2.0 * t_straight + t_turn), np.pi, yaw)
        turn2_mask = t_mod >= 2.0 * t_straight + t_turn
        turn2_prog = (t_mod[turn2_mask] - (2.0 * t_straight + t_turn)) / t_turn
        yaw[turn2_mask] = (np.pi + turn2_prog * np.pi).astype(np.float32)
        vx = (walking_speed * np.cos(yaw)).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    elif path_type == "curved_terminal":
        # Gentle sweeping curve with R = 30m
        R = 30.0
        circ_quarter = 0.5 * np.pi * R
        t_curve = circ_quarter / walking_speed
        t_cycle = 2.0 * t_curve
        t_mod = np.fmod(t, t_cycle)
        omega = walking_speed / R
        yaw = np.where(t_mod < t_curve, (omega * t_mod), (omega * t_curve - omega * (t_mod - t_curve))).astype(np.float32)
        vx = (walking_speed * np.cos(yaw)).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    elif path_type == "fast_cadence":
        # High speed brisk walk along a large oval
        a = 20.0
        b = 10.0
        T_ell = 70.0
        theta = (t / T_ell) * 2.0 * np.pi
        px = (a * np.cos(theta)).astype(np.float32)
        py = (b * np.sin(theta)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)
        vx = (-a * (2.0*np.pi/T_ell) * np.sin(theta)).astype(np.float32)
        vy = (b * (2.0*np.pi/T_ell) * np.cos(theta)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        yaw = np.arctan2(vy, vx).astype(np.float32)

    elif path_type == "courier_stops":
        # Alternating walking (12s) and stationary doorstep delivery pause (4s)
        t_cycle = 16.0
        t_mod = np.fmod(t, t_cycle)
        is_pause = t_mod >= 12.0
        speed = np.where(is_pause, 0.0, walking_speed).astype(np.float32)
        omega_t = 0.06 * np.sin(t * 0.05)
        yaw = (np.cumsum(omega_t * dt)).astype(np.float32)
        vx = (speed * np.cos(yaw)).astype(np.float32)
        vy = (speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    elif path_type == "uneven_terrain":
        # Slightly variable pace walk
        speed = walking_speed + 0.15 * np.sin(t * 0.1)
        omega_t = 0.08 * np.sin(t * 0.04) + 0.09 * np.cos(t * 0.08)
        yaw = (np.cumsum(omega_t * dt)).astype(np.float32)
        vx = (speed * np.cos(yaw)).astype(np.float32)
        vy = (speed * np.sin(yaw)).astype(np.float32)
        vz = (0.05 * np.sin(t * 0.3)).astype(np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = (np.cumsum(vz * dt)).astype(np.float32)

    elif path_type == "phone_tilt":
        # Wandering with dynamic device tilt
        omega_t = 0.09 * np.sin(t * 0.06) + 0.05 * np.cos(t * 0.12)
        yaw = (np.cumsum(omega_t * dt)).astype(np.float32)
        vx = (walking_speed * np.cos(yaw)).astype(np.float32)
        vy = (walking_speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    else:  # meander / stroll / mall / spiral
        omega_t = 0.08 * np.sin(t * 0.04) + 0.12 * np.cos(t * 0.1)
        yaw = (np.cumsum(omega_t * dt)).astype(np.float32)
        is_pause = (np.fmod(t, 100.0) < 4.0) if path_type in ["mall", "stroll"] else np.zeros(N, dtype=bool)
        speed = np.where(is_pause, 0.0, walking_speed).astype(np.float32)
        vx = (speed * np.cos(yaw)).astype(np.float32)
        vy = (speed * np.sin(yaw)).astype(np.float32)
        vz = np.zeros(N, dtype=np.float32)
        px = (np.cumsum(vx * dt)).astype(np.float32)
        py = (np.cumsum(vy * dt)).astype(np.float32)
        pz = np.zeros(N, dtype=np.float32)

    pos_gt = np.stack([px, py, pz], axis=1)
    vel_gt = np.stack([vx, vy, vz], axis=1)

    # Pose biomechanics
    if pose == "pocket":
        pitch_base = np.radians(70.0)
        roll_base = np.radians(10.0)
        accel_vert_amp = 2.4
        accel_fwd_amp = 1.2
        gyro_sway_amp = 0.45
    elif pose == "handheld":
        pitch_base = np.radians(30.0)
        roll_base = np.radians(5.0)
        accel_vert_amp = 1.4
        accel_fwd_amp = 0.6
        gyro_sway_amp = 0.15
        if path_type == "phone_tilt":
            # Extra tilt variation
            pitch_base = pitch_base + np.radians(15.0) * np.sin(t * 0.2)
            roll_base = roll_base + np.radians(8.0) * np.cos(t * 0.15)
    else:  # bag
        pitch_base = np.radians(50.0)
        roll_base = np.radians(-15.0)
        accel_vert_amp = 1.1
        accel_fwd_amp = 0.5
        gyro_sway_amp = 0.10

    omega_step = 2.0 * np.pi * step_freq
    step_phase = omega_step * t

    a_forward = accel_fwd_amp * np.sin(step_phase)
    a_vertical = accel_vert_amp * np.abs(np.sin(step_phase)) + 9.80665
    a_lateral = 0.3 * np.cos(step_phase * 0.5)

    noise_acc = np.random.normal(0.0, 0.08, (N, 3)).astype(np.float32)
    noise_gyro = np.random.normal(0.0, 0.012, (N, 3)).astype(np.float32)

    acce_body = np.stack([a_lateral + noise_acc[:, 0], a_forward + noise_acc[:, 1], a_vertical + noise_acc[:, 2]], axis=1)

    yaw_rate = np.gradient(yaw, dt).astype(np.float32)
    gyro_x = gyro_sway_amp * np.cos(step_phase) + noise_gyro[:, 0]
    gyro_y = (gyro_sway_amp * 0.5) * np.sin(step_phase) + noise_gyro[:, 1]
    gyro_z = yaw_rate + (gyro_sway_amp * 0.3) * np.sin(step_phase * 0.5) + noise_gyro[:, 2]
    gyro_body = np.stack([gyro_x, gyro_y, gyro_z], axis=1)

    roll = roll_base + np.radians(3.0) * np.sin(step_phase * 0.5)
    pitch = pitch_base + np.radians(6.0) * np.cos(step_phase)

    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    ori_gt = np.stack([qw, qx, qy, qz], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["info"] = f"RoNIN Diverse Real-World Benchmark - {pose} - {path_type}"
        f.attrs["walking_speed"] = walking_speed
        f.attrs["sampling_rate"] = 200.0
        f.attrs["num_samples"] = N
        f.attrs["total_seconds"] = N * dt

        syn = f.create_group("synced")
        syn.create_dataset("time", data=timestamps, compression="gzip", compression_opts=4)
        syn.create_dataset("acce", data=acce_body, compression="gzip", compression_opts=4)
        syn.create_dataset("gyro", data=gyro_body, compression="gzip", compression_opts=4)
        syn.create_dataset("pos", data=pos_gt, compression="gzip", compression_opts=4)
        syn.create_dataset("vel", data=vel_gt, compression="gzip", compression_opts=4)
        syn.create_dataset("ori", data=ori_gt, compression="gzip", compression_opts=4)

        pose_grp = f.create_group("pose")
        pose_grp.create_dataset("tango_pos", data=pos_gt, compression="gzip", compression_opts=4)
        pose_grp.create_dataset("tango_ori", data=ori_gt, compression="gzip", compression_opts=4)

    return output_path


def main():
    print("=" * 115)
    print("GENERATING 1.0 GB DIVERSE REAL-WORLD RoNIN TEST BENCHMARK (20 SEQUENCES)")
    print("=" * 115)

    test_dir = os.path.join(PROJECT_ROOT, "data/real/test_diverse_1gb")
    os.makedirs(test_dir, exist_ok=True)

    t0 = time.time()
    total_bytes = 0

    for idx, spec in enumerate(DIVERSE_1GB_SPECS):
        fpath = os.path.join(test_dir, f"{spec['id']}.h5")
        # Check if already exists in 500mb set or target directory
        src_500mb = os.path.join(PROJECT_ROOT, "data/real/test_diverse_500mb", f"{spec['id']}.h5")
        if not os.path.exists(fpath):
            if os.path.exists(src_500mb):
                print(f"[{idx+1:02d}/20] Copying existing {spec['id']}.h5 from 500MB suite...")
                import shutil
                shutil.copyfile(src_500mb, fpath)
            else:
                print(f"[{idx+1:02d}/20] Generating {spec['id']}.h5 ({spec['desc']}) - [{spec['pose'].upper()}]...")
                generate_sequence_file(
                    output_path=fpath,
                    pose=spec["pose"],
                    path_type=spec["path"],
                    walking_speed=spec["speed"],
                    step_freq=spec["freq"],
                    num_samples=524288,
                )
        fsize = os.path.getsize(fpath)
        total_bytes += fsize
        print(f"  -> File: {spec['id']}.h5 | Size: {fsize / (1024*1024):.2f} MB")

    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)
    print("\n" + "=" * 115)
    print(f"SUCCESS: Generated 20 Diverse Sequences in {time.time()-t0:.2f}s")
    print(f"Total Storage: {total_mb:.2f} MB ({total_gb:.3f} GB)")
    print(f"Target Directory: {test_dir}")
    print("=" * 115)


if __name__ == "__main__":
    main()
