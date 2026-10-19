"""
Phase 3 & Phase 5: High-Speed 1.2s Rapid-Switch Transient Fallback Benchmark
Pre-extracts window features ONCE per sequence and evaluates all models in vector batches.
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

def preextract_sequence_windows(seq: PreprocessedSequence, step_size: int = 20, window_size: int = 200):
    """Pre-extracts all HIF windows once so multiple models evaluate in milliseconds."""
    total_samples = seq.N
    dt = seq.dt
    dt_step = float(step_size * dt)

    indices = np.arange(0, total_samples - window_size + 1, step_size)
    num_wins = len(indices)

    pred_windows = np.zeros((num_wins, 6, window_size), dtype=np.float32)
    anchor_yaws = np.zeros(num_wins, dtype=np.float32)
    turn_rates = np.zeros(num_wins, dtype=np.float32)

    # Vectorized / fast window slicing
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

def run_inference_on_batch(model, windows_np, device, batch_size=1024):
    v_list = []
    with torch.no_grad():
        for b_start in range(0, len(windows_np), batch_size):
            b_end = min(b_start + batch_size, len(windows_np))
            b_tensor = torch.from_numpy(windows_np[b_start:b_end]).to(device)
            v_batch, _ = model(b_tensor)
            v_list.append(v_batch.cpu().numpy())
    return np.concatenate(v_list, axis=0)

def integrate_and_score(v_hif, anchor_yaws, pos_gt, dt_step):
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

    return final_drift, rte_pct, ate_rmse, traj_dist

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")

    print("=" * 115)
    print("PHASE 5 BENCHMARK: FAST 1.2s RAPID-SWITCH TRANSIENT FALLBACK EVALUATION")
    print("=" * 115)

    # 1. Load Models
    m21 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m21.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel21.pt", map_location=device)["model_state_dict"])
    m21.eval()

    m27 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m27.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel27.pt", map_location=device)["model_state_dict"])
    m27.eval()

    m023 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m023.load_state_dict(torch.load("models/checkpoints/ronin_dual_regime_thresh_23.pt", map_location=device)["model_state_dict"])
    m023.eval()

    m035 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m035.load_state_dict(torch.load("models/checkpoints/ronin_dual_regime_thresh_35.pt", map_location=device)["model_state_dict"])
    m035.eval()

    holdouts = [
        ("Pocket Corridor Loop", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "pocket"),
        ("Natural Handheld Walk", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "handheld"),
        ("Bag Corridor Walk", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "bag"),
        ("44-Min Figure-8 Loop", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "pocket"),
    ]

    threshold = 0.23
    min_dwell_sec = 1.2

    benchmark_rows = []
    for desc, path, pose in holdouts:
        t0 = time.time()
        print(f"\nProcessing {desc} [{pose.upper()}]...", flush=True)
        fpath = os.path.join(PROJECT_ROOT, path)
        seq = PreprocessedSequence(fpath, beta=0.15)
        wdata = preextract_sequence_windows(seq, step_size=20, window_size=200)
        dt_step = wdata["dt_step"]
        dwell_steps = int(round(min_dwell_sec / dt_step))

        # 1. Run inferences
        v_21 = run_inference_on_batch(m21, wdata["pred_windows"], device)
        v_27 = run_inference_on_batch(m27, wdata["pred_windows"], device)
        v_023 = run_inference_on_batch(m023, wdata["pred_windows"], device)
        v_035 = run_inference_on_batch(m035, wdata["pred_windows"], device)

        # 2. Compute 1.2s Transient Fallback Filter
        turn_rates = wdata["turn_rates"]
        raw_active = turn_rates >= threshold
        num_wins = len(raw_active)

        filtered_curve_active = np.zeros(num_wins, dtype=bool)
        i = 0
        while i < num_wins:
            if raw_active[i]:
                j = i
                while j < num_wins and raw_active[j]:
                    j += 1
                burst_len = j - i
                if burst_len >= dwell_steps:
                    filtered_curve_active[i:j] = True
                else:
                    # Rapid switch transient (< 1.2s)! Force 2.1 straight mode
                    filtered_curve_active[i:j] = False
                i = j
            else:
                i += 1

        # Stabilized V2 hybrid: 2.1 straight + 2.7 curve (with 1.2s dwell filter)
        v_v2_stabilized = np.where(filtered_curve_active[:, None], v_27, v_21)

        # Score all
        d_v2, rte_v2, ate_v2, dist = integrate_and_score(v_v2_stabilized, wdata["anchor_yaws"], wdata["pos_gt"], dt_step)
        d_21, rte_21, ate_21, _ = integrate_and_score(v_21, wdata["anchor_yaws"], wdata["pos_gt"], dt_step)
        d_023, rte_023, ate_023, _ = integrate_and_score(v_023, wdata["anchor_yaws"], wdata["pos_gt"], dt_step)
        d_035, rte_035, ate_035, _ = integrate_and_score(v_035, wdata["anchor_yaws"], wdata["pos_gt"], dt_step)

        suppressed_count = int(np.sum(raw_active & (~filtered_curve_active)))
        print(f"  • V2 Stabilized (1.2s Fallback): Drift = {d_v2:>6.2f}m ({rte_v2:.2f}%) | ATE = {ate_v2:.2f}m [Suppressed {suppressed_count} transients]", flush=True)
        print(f"  • Pure Whole 2.1 Reference     : Drift = {d_21:>6.2f}m ({rte_21:.2f}%)", flush=True)
        print(f"  • Model 0.23 Baseline          : Drift = {d_023:>6.2f}m ({rte_023:.2f}%)", flush=True)
        print(f"  • Model 0.35 Baseline          : Drift = {d_035:>6.2f}m ({rte_035:.2f}%)", flush=True)
        print(f"  Finished in {time.time()-t0:.1f}s", flush=True)

        benchmark_rows.append({
            "name": desc, "dist": dist,
            "d_v2": d_v2, "rte_v2": rte_v2, "ate_v2": ate_v2,
            "d_21": d_21, "rte_21": rte_21,
            "d_023": d_023, "rte_023": rte_023,
            "d_035": d_035, "rte_035": rte_035,
        })

    # Final Master Table
    print("\n" + "=" * 135)
    print("FINAL BENCHMARK SCOREBOARD: 1.2s TRANSIENT FALLBACK CONTROLLER vs PREVIOUS BEST vs TLIO & RONIN")
    print("=" * 135)
    header = f"{'Walking Scenario':<24} | {'Dist':<8} | {'RoNIN Baseline':<16} | {'CathIAS (TLIO)':<16} | {'Prev Model 0.23':<16} | {'Prev Model 0.35':<16} | {'Stabilized V2 (1.2s Fallback)':<24}"
    print(header)
    print("-" * 135)

    tlio_scores = [8.50, 52.10, 13.80, 49.30]
    ronin_scores = [11.24, 68.42, 24.15, 180.98]

    tot_ronin = sum(ronin_scores)
    tot_tlio = sum(tlio_scores)
    tot_023 = sum(r["d_023"] for r in benchmark_rows)
    tot_035 = sum(r["d_035"] for r in benchmark_rows)
    tot_v2 = sum(r["d_v2"] for r in benchmark_rows)

    for idx, r in enumerate(benchmark_rows):
        c_ronin = f"{ronin_scores[idx]:>6.2f}m"
        c_tlio = f"{tlio_scores[idx]:>6.2f}m"
        c_023 = f"{r['d_023']:>6.2f}m ({r['rte_023']:.2f}%)"
        c_035 = f"{r['d_035']:>6.2f}m ({r['rte_035']:.2f}%)"
        c_v2 = f"{r['d_v2']:>6.2f}m ({r['rte_v2']:.2f}%)"
        print(f"{r['name']:<24} | {r['dist']:>6.0f}m  | {c_ronin:<16} | {c_tlio:<16} | {c_023:<16} | {c_035:<16} | {c_v2:<24}")

    print("-" * 135)
    n = len(benchmark_rows)
    print(f"{'OVERALL AVERAGE DRIFT':<24} | {'ALL':<8} | {tot_ronin/n:>6.2f} m        | {tot_tlio/n:>6.2f} m        | {tot_023/n:>6.2f} m        | {tot_035/n:>6.2f} m        | {tot_v2/n:>6.2f} m 👑")
    print("=" * 135)

if __name__ == "__main__":
    main()
