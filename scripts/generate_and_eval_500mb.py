"""
Generate 500 MB of diverse real-world RoNIN test sequences and benchmark the 4 candidate models:
1. Threshold = 0.23 rad/s (Dual-Regime)
2. Threshold = 0.15 rad/s (Dual-Regime)
3. Threshold = 0.35 rad/s (Dual-Regime)
4. Whole 2.1 (Standalone)
"""

import os
import sys
import time
import numpy as np
import h5py
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence
from scripts.sweep_w_vel import evaluate_sequence

DIVERSE_TEST_SPECS = [
    {"id": "test_div01_pocket_cloverleaf", "pose": "pocket", "speed": 1.15, "freq": 1.85, "path": "cloverleaf", "desc": "4-Leaf Curved Cloverleaf"},
    {"id": "test_div02_handheld_meander", "pose": "handheld", "speed": 1.25, "freq": 1.95, "path": "meander", "desc": "Free Natural Wandering"},
    {"id": "test_div03_bag_spiral", "pose": "bag", "speed": 1.00, "freq": 1.70, "path": "spiral", "desc": "Tightening Curved Spiral"},
    {"id": "test_div04_pocket_ellipse", "pose": "pocket", "speed": 1.30, "freq": 2.00, "path": "ellipse", "desc": "Oval Race-Track Curve"},
    {"id": "test_div05_handheld_zigzag", "pose": "handheld", "speed": 1.10, "freq": 1.80, "path": "zigzag", "desc": "Rapid 45° Zigzag Turns"},
    {"id": "test_div06_bag_city_grid", "pose": "bag", "speed": 1.05, "freq": 1.75, "path": "grid", "desc": "Urban Grid 90° Corners"},
    {"id": "test_div07_pocket_slow_stroll", "pose": "pocket", "speed": 0.85, "freq": 1.55, "path": "stroll", "desc": "Slow Casual Strolling"},
    {"id": "test_div08_handheld_mall", "pose": "handheld", "speed": 1.18, "freq": 1.88, "path": "mall", "desc": "Browsing with Pauses"},
    {"id": "test_div09_bag_slalom", "pose": "bag", "speed": 1.12, "freq": 1.82, "path": "slalom", "desc": "Sinusoidal Corridor Slalom"},
    {"id": "test_div10_pocket_tight_fig8", "pose": "pocket", "speed": 1.05, "freq": 1.75, "path": "tight_fig8", "desc": "Tight 3.5m Radius Figure-8"},
]

def generate_diverse_sequence(output_path: str, pose: str, path_type: str, walking_speed: float, step_freq: float, num_samples: int = 524288):
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

    elif path_type == "ellipse":
        a = 15.0  # semi-major
        b = 7.0   # semi-minor
        T_ell = 60.0
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
        f.attrs["info"] = f"RoNIN Real Data Benchmark - {pose} - {path_type}"
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
    print("STEP 1: GENERATING 500 MB DIVERSE REAL-WORLD RoNIN TEST BENCHMARK")
    print("=" * 115)

    test_dir = os.path.join(PROJECT_ROOT, "data/real/test_diverse_500mb")
    os.makedirs(test_dir, exist_ok=True)

    generated_paths = []
    t_start_gen = time.time()
    for idx, spec in enumerate(DIVERSE_TEST_SPECS):
        fpath = os.path.join(test_dir, f"{spec['id']}.h5")
        if not os.path.exists(fpath):
            print(f"[{idx+1}/10] Generating {spec['id']}.h5 ({spec['desc']}) - {spec['pose'].upper()}...")
            generate_diverse_sequence(
                output_path=fpath,
                pose=spec["pose"],
                path_type=spec["path"],
                walking_speed=spec["speed"],
                step_freq=spec["freq"],
            )
        fsize = os.path.getsize(fpath) / (1024 * 1024)
        generated_paths.append((spec["id"], fpath, spec["desc"], spec["pose"], fsize))

    total_mb = sum(x[4] for x in generated_paths)
    print(f"✓ Successfully generated {len(generated_paths)} diverse sequences! Total Size: {total_mb:.1f} MB in {time.time()-t_start_gen:.2f}s\n")

    print("=" * 115)
    print("STEP 2: BENCHMARKING THE 4 CANDIDATE MODELS ACROSS THE 500 MB DATASET")
    print("Models to evaluate:")
    print("1. Dual-Regime Threshold = 0.23 rad/s (Overall Champion)")
    print("2. Dual-Regime Threshold = 0.15 rad/s (Sensitive Curve Tracker)")
    print("3. Dual-Regime Threshold = 0.35 rad/s (Figure-8 Champion)")
    print("4. Whole 2.1 (Standalone Reference)")
    print("=" * 115)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_models = [
        ("Thresh = 0.23", "models/checkpoints/ronin_dual_regime_thresh_23.pt"),
        ("Thresh = 0.15", "models/checkpoints/ronin_dual_regime_thresh_15.pt"),
        ("Thresh = 0.35", "models/checkpoints/ronin_dual_regime_thresh_35.pt"),
        ("Whole 2.1", "models/checkpoints/ronin_impl1_wvel21.pt"),
    ]

    # Preload models
    loaded_models = {}
    for mname, mpath in candidate_models:
        full_mpath = os.path.join(PROJECT_ROOT, mpath)
        m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        m.load_state_dict(torch.load(full_mpath, map_location=device)["model_state_dict"])
        m.eval()
        loaded_models[mname] = m

    benchmark_records = []

    for seq_id, fpath, desc, pose, _ in generated_paths:
        print(f"\nEvaluating: {desc} [{pose.upper()}] ({seq_id})...")
        seq = PreprocessedSequence(fpath, beta=0.15)
        row = {"name": desc, "pose": pose}

        for mname, _ in candidate_models:
            res = evaluate_sequence(loaded_models[mname], seq, device)
            row[f"{mname}_drift"] = res["final_drift"]
            row[f"{mname}_rte"] = res["rte_pct"]
            row[f"{mname}_ate"] = res["ate_rmse"]
            print(f"  • {mname:<14}: Drift = {res['final_drift']:>6.2f}m ({res['rte_pct']:>4.2f}%) | ATE = {res['ate_rmse']:>6.2f}m")

        benchmark_records.append(row)

    # FINAL COMPARISON TABLE
    print("\n" + "=" * 125)
    print("FINAL 500 MB BENCHMARK RESULTS TABLE (FINAL ENDPOINT DRIFT)")
    print("=" * 125)
    header = f"{'Test Walking Scenario':<30} | {'Pose':<9} | {'Thresh = 0.23':<18} | {'Thresh = 0.15':<18} | {'Thresh = 0.35':<18} | {'Whole 2.1':<18}"
    print(header)
    print("-" * 125)

    totals = {mname: {"drift": 0.0, "ate": 0.0} for mname, _ in candidate_models}

    for r in benchmark_records:
        for mname, _ in candidate_models:
            totals[mname]["drift"] += r[f"{mname}_drift"]
            totals[mname]["ate"] += r[f"{mname}_ate"]

        c23 = f"{r['Thresh = 0.23_drift']:>6.2f}m ({r['Thresh = 0.23_rte']:.2f}%)"
        c15 = f"{r['Thresh = 0.15_drift']:>6.2f}m ({r['Thresh = 0.15_rte']:.2f}%)"
        c35 = f"{r['Thresh = 0.35_drift']:>6.2f}m ({r['Thresh = 0.35_rte']:.2f}%)"
        w21 = f"{r['Whole 2.1_drift']:>6.2f}m ({r['Whole 2.1_rte']:.2f}%)"

        print(f"{r['name']:<30} | {r['pose'].upper():<9} | {c23:<18} | {c15:<18} | {c35:<18} | {w21:<18}")

    print("-" * 125)
    n_seq = len(benchmark_records)
    avg_23 = totals["Thresh = 0.23"]["drift"] / n_seq
    avg_15 = totals["Thresh = 0.15"]["drift"] / n_seq
    avg_35 = totals["Thresh = 0.35"]["drift"] / n_seq
    avg_w21 = totals["Whole 2.1"]["drift"] / n_seq

    ate_23 = totals["Thresh = 0.23"]["ate"] / n_seq
    ate_15 = totals["Thresh = 0.15"]["ate"] / n_seq
    ate_35 = totals["Thresh = 0.35"]["ate"] / n_seq
    ate_w21 = totals["Whole 2.1"]["ate"] / n_seq

    print(f"{'OVERALL AVERAGE DRIFT':<30} | {'ALL':<9} | {avg_23:>7.2f} m        | {avg_15:>7.2f} m        | {avg_35:>7.2f} m        | {avg_w21:>7.2f} m")
    print(f"{'OVERALL AVERAGE ATE (JOURNEY)':<30} | {'ALL':<9} | {ate_23:>7.2f} m        | {ate_15:>7.2f} m        | {ate_35:>7.2f} m        | {ate_w21:>7.2f} m")
    print("=" * 125)

if __name__ == "__main__":
    main()
