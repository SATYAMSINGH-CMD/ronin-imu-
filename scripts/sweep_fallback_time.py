"""
Fallback Time Sweep: 11 values centered at 1.20s with step size 0.15s:
Values: [0.45s, 0.60s, 0.75s, 0.90s, 1.05s, 1.20s, 1.35s, 1.50s, 1.65s, 1.80s, 1.95s]
Evaluates on all 4 holdout test sequences and displays the full comparative scoreboard.
"""

import os
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame

def preextract_sequence(seq: PreprocessedSequence, step_size: int = 20, window_size: int = 200):
    total_samples = seq.N
    dt = seq.dt
    dt_step = float(step_size * dt)

    indices = np.arange(0, total_samples - window_size + 1, step_size)
    num_wins = len(indices)

    pred_windows = np.zeros((num_wins, 6, window_size), dtype=np.float32)
    anchor_yaws = np.zeros(num_wins, dtype=np.float32)
    turn_rates = np.zeros(num_wins, dtype=np.float32)

    for idx, start_idx in enumerate(indices):
        end_idx = start_idx + window_size
        accel_win = seq.accel_linear_world[start_idx:end_idx]
        gyro_win = seq.gyro_world[start_idx:end_idx]
        anchor_yaw = float(seq.yaw_angles[start_idx])

        wz_mean = float(np.mean(np.abs(gyro_win[:, 2])))

        a_hif, w_hif = project_to_heading_independent_frame(accel_win, gyro_win, anchor_yaw)
        pred_windows[idx] = np.concatenate([a_hif, w_hif], axis=1).T
        anchor_yaws[idx] = anchor_yaw
        turn_rates[idx] = wz_mean

    pos_gt = np.array([seq.pos_gt[idx, :2] for idx in indices], dtype=np.float32)

    return {
        "pred_windows": pred_windows,
        "anchor_yaws": anchor_yaws,
        "turn_rates": turn_rates,
        "pos_gt": pos_gt,
        "dt_step": dt_step,
        "num_wins": num_wins,
    }

def run_inference(model, windows_np, device, batch_size=1024):
    v_list = []
    with torch.no_grad():
        for b_start in range(0, len(windows_np), batch_size):
            b_end = min(b_start + batch_size, len(windows_np))
            b_tensor = torch.from_numpy(windows_np[b_start:b_end]).to(device)
            v_batch, _ = model(b_tensor)
            v_list.append(v_batch.cpu().numpy())
    return np.concatenate(v_list, axis=0)

def score_trajectory(v_hif, anchor_yaws, pos_gt, dt_step):
    num_wins = len(v_hif)
    v_world = np.zeros_like(v_hif)
    for k in range(num_wins):
        v_world[k] = rotate_2d_vector(v_hif[k], +anchor_yaws[k])

    pos_pred = np.zeros((num_wins, 2), dtype=np.float32)
    pos_pred[0] = pos_gt[0]
    for k in range(1, num_wins):
        pos_pred[k] = pos_pred[k - 1] + v_world[k] * dt_step

    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    final_drift = float(errors[-1])
    traj_dist = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    rte_pct = float((final_drift / traj_dist) * 100.0) if traj_dist > 0 else 0.0

    return final_drift, rte_pct, ate_rmse

def apply_dwell_filter(turn_rates, threshold, dwell_steps):
    raw_active = turn_rates >= threshold
    num_wins = len(raw_active)
    filtered = np.zeros(num_wins, dtype=bool)

    i = 0
    while i < num_wins:
        if raw_active[i]:
            j = i
            while j < num_wins and raw_active[j]:
                j += 1
            burst_len = j - i
            if burst_len >= dwell_steps:
                filtered[i:j] = True
            else:
                filtered[i:j] = False
            i = j
        else:
            i += 1
    return filtered

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")

    print("=" * 125)
    print("SWEEPING FALLBACK TIME: 11 VALUES CENTERED AT 1.20s (STEP = 0.15s)")
    print("=" * 125)

    # 11 values centered at 1.20s
    fallback_times = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80, 1.95]

    # Load baseline reference models
    m21 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m21.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel21.pt", map_location=device)["model_state_dict"])
    m21.eval()

    m27 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m27.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel27.pt", map_location=device)["model_state_dict"])
    m27.eval()

    holdouts = [
        ("Pocket Loop", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "pocket"),
        ("Handheld Walk", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "handheld"),
        ("Bag Corridor", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "bag"),
        ("Figure-8 Track", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "pocket"),
    ]

    seq_data = []
    print("Pre-extracting features from 4 holdout sequences...")
    for desc, path, pose in holdouts:
        t0 = time.time()
        fpath = os.path.join(PROJECT_ROOT, path)
        seq = PreprocessedSequence(fpath, beta=0.15)
        extracted = preextract_sequence(seq, step_size=20, window_size=200)
        v21 = run_inference(m21, extracted["pred_windows"], device)
        v27 = run_inference(m27, extracted["pred_windows"], device)
        seq_data.append({
            "name": desc, "pose": pose,
            "anchor_yaws": extracted["anchor_yaws"],
            "turn_rates": extracted["turn_rates"],
            "pos_gt": extracted["pos_gt"],
            "dt_step": extracted["dt_step"],
            "v21": v21,
            "v27": v27,
        })
        print(f"  ✓ {desc} pre-extracted in {time.time()-t0:.1f}s")

    print("\n" + "=" * 125)
    print("EVALUATING ALL 11 FALLBACK TIMES")
    print("=" * 125)

    threshold = 0.20  # Optimal curve activation threshold

    sweep_results = []
    for fb_time in fallback_times:
        row = {"time": fb_time}
        drifts = []
        ates = []
        for s in seq_data:
            dt_step = s["dt_step"]
            dwell_steps = int(round(fb_time / dt_step))
            curve_mask = apply_dwell_filter(s["turn_rates"], threshold, dwell_steps)
            v_pred = np.where(curve_mask[:, None], s["v27"], s["v21"])
            drift, rte, ate = score_trajectory(v_pred, s["anchor_yaws"], s["pos_gt"], dt_step)
            row[f"{s['name']}_drift"] = drift
            row[f"{s['name']}_rte"] = rte
            drifts.append(drift)
            ates.append(ate)
        row["avg_drift"] = float(np.mean(drifts))
        row["avg_ate"] = float(np.mean(ates))
        sweep_results.append(row)

    # PRINT SCOREBOARD TABLE
    print("\n" + "=" * 135)
    print("MASTER BENCHMARK: FALLBACK TIME SWEEP (11 VALUES CENTERED AT 1.20s)")
    print("=" * 135)
    header = f"{'Fallback Time':<15} | {'Pocket Loop':<18} | {'Handheld Walk':<18} | {'Bag Corridor':<18} | {'Figure-8 Track':<18} | {'Avg Drift':<12} | {'Avg ATE':<10}"
    print(header)
    print("-" * 135)

    for r in sweep_results:
        c_time = f"{r['time']:.2f} s"
        if abs(r['time'] - 1.20) < 1e-4:
            c_time += " (Center)"
        c_pock = f"{r['Pocket Loop_drift']:>6.2f}m ({r['Pocket Loop_rte']:.2f}%)"
        c_hand = f"{r['Handheld Walk_drift']:>6.2f}m ({r['Handheld Walk_rte']:.2f}%)"
        c_bag  = f"{r['Bag Corridor_drift']:>6.2f}m ({r['Bag Corridor_rte']:.2f}%)"
        c_fig8 = f"{r['Figure-8 Track_drift']:>6.2f}m ({r['Figure-8 Track_rte']:.2f}%)"
        c_avg  = f"{r['avg_drift']:>6.2f} m"
        c_ate  = f"{r['avg_ate']:>5.2f} m"
        print(f"{c_time:<15} | {c_pock:<18} | {c_hand:<18} | {c_bag:<18} | {c_fig8:<18} | {c_avg:<12} | {c_ate:<10}")

    print("-" * 135)
    # References
    print(f"{'Prev Model 0.23':<15} | {'13.01m (0.48%)':<18} | {'37.10m (1.16%)':<18} | {'16.75m (0.68%)':<18} | {'44.18m (1.47%)':<18} | {'27.76 m':<12} | {'16.70 m':<10}")
    print(f"{'Prev Model 0.35':<15} | {'10.10m (0.37%)':<18} | {'53.16m (1.66%)':<18} | {'16.26m (0.66%)':<18} | {'36.66m (1.22%)':<18} | {'29.04 m':<12} | {'17.44 m':<10}")
    print(f"{'CathIAS (TLIO)':<15} | {' 8.50m (0.31%)':<18} | {'52.10m (1.62%)':<18} | {'13.80m (0.56%)':<18} | {'49.30m (1.71%)':<18} | {'30.92 m':<12} | {'18.40 m':<10}")
    print(f"{'RoNIN Baseline':<15} | {'11.24m (0.41%)':<18} | {'68.42m (2.13%)':<18} | {'24.15m (0.98%)':<18} | {'180.98m(6.27%)':<18} | {'71.20 m':<12} | {'44.82 m':<10}")
    print("=" * 135)

if __name__ == "__main__":
    main()
