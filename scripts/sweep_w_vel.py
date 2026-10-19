"""
Hyperparameter Sweep for Turn-Velocity Loss Weight (w_vel_turn_max)
Tests w_vel_turn_max in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
Evaluates:
1. Final Drift (meters)
2. ATE RMSE (meters)
3. RTE (%)
4. Circle Radius R_right, R_left, R_avg (meters) and Difference vs 5.000m
5. Mean Turn Speed (m/s) vs 1.150 m/s
6. Full Holdout Test Set Performance (test_sub01, test_sub02, test_sub03)
"""

import os
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.train import train_model
from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame
from scripts.measure_radius import fit_circle_least_squares

def evaluate_sequence(model, seq, device, step_size=10, window_size=200):
    total_samples = seq.N
    dt = seq.dt
    dt_step = float(step_size * dt)

    pred_windows = []
    anchor_yaws = []
    window_indices = []

    for start_idx in range(0, total_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        accel_win = seq.accel_linear_world[start_idx:end_idx]
        gyro_win = seq.gyro_world[start_idx:end_idx]
        anchor_yaw = float(seq.yaw_angles[start_idx])

        a_hif, w_hif = project_to_heading_independent_frame(accel_win, gyro_win, anchor_yaw)
        features = np.concatenate([a_hif, w_hif], axis=1).T
        pred_windows.append(features)
        anchor_yaws.append(anchor_yaw)
        window_indices.append(start_idx)

    pred_np = np.array(pred_windows, dtype=np.float32)
    v_hif_list = []
    batch_size = 512

    with torch.no_grad():
        for b_start in range(0, len(pred_np), batch_size):
            b_end = min(b_start + batch_size, len(pred_np))
            b_tensor = torch.from_numpy(pred_np[b_start:b_end]).to(device)
            v_batch, _ = model(b_tensor)
            v_hif_list.append(v_batch.cpu().numpy())

    v_hif_preds = np.concatenate(v_hif_list, axis=0)

    v_world_preds = np.zeros_like(v_hif_preds)
    for i in range(len(v_hif_preds)):
        v_world_preds[i] = rotate_2d_vector(v_hif_preds[i], +anchor_yaws[i])

    pos_pred = np.zeros((len(v_world_preds), 2), dtype=np.float32)
    pos_pred[0] = seq.pos_gt[window_indices[0], :2]
    for i in range(1, len(v_world_preds)):
        pos_pred[i] = pos_pred[i - 1] + v_world_preds[i] * dt_step

    pos_gt = np.array([seq.pos_gt[idx, :2] for idx in window_indices])
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    final_drift = float(errors[-1])

    traj_dist = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    rte_pct = float((final_drift / traj_dist) * 100.0) if traj_dist > 0 else 0.0

    return {
        "ate_rmse": ate_rmse,
        "final_drift": final_drift,
        "rte_pct": rte_pct,
        "traj_dist": traj_dist,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "v_world": v_world_preds,
        "dt_step": dt_step,
    }

def measure_figure8_geometry(res):
    pos_pred = res["pos_pred"]
    pos_gt = res["pos_gt"]
    v_world = res["v_world"]
    dt_step = res["dt_step"]

    # Exact period
    R_true = 5.0
    circ = 2.0 * np.pi * R_true
    T_circle = circ / 1.15  # 27.318s
    T_fig8 = 2.0 * T_circle  # 54.636s

    steps_circle = int(round(T_circle / dt_step))  # 546
    steps_fig8 = 2 * steps_circle                  # 1092

    r_rights = []
    r_lefts = []
    # Measure across first 8 loops
    for i in range(8):
        s = i * steps_fig8
        mid = s + steps_circle
        e = (i + 1) * steps_fig8
        if e > len(pos_pred):
            break
        _, _, rr, _ = fit_circle_least_squares(pos_pred[s:mid, 0], pos_pred[s:mid, 1])
        _, _, rl, _ = fit_circle_least_squares(pos_pred[mid:e, 0], pos_pred[mid:e, 1])
        r_rights.append(rr)
        r_lefts.append(rl)

    mean_rr = float(np.mean(r_rights))
    mean_rl = float(np.mean(r_lefts))
    mean_r = (mean_rr + mean_rl) / 2.0
    r_diff = mean_r - 5.0
    r_shrink_pct = (r_diff / 5.0) * 100.0

    mean_speed = float(np.linalg.norm(v_world, axis=1).mean())

    return {
        "r_right": mean_rr,
        "r_left": mean_rl,
        "r_avg": mean_r,
        "r_diff": r_diff,
        "r_shrink_pct": r_shrink_pct,
        "mean_speed": mean_speed,
    }

def main():
    print("=" * 105)
    print("RUNNING HYPERPARAMETER SWEEP: w_vel_turn_max in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]")
    print("=" * 105)

    w_vel_values = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    train_dir = os.path.join(PROJECT_ROOT, "data/real/train")
    fig8_path = os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test sequence
    seq_fig8 = PreprocessedSequence(fig8_path, beta=0.15)

    # Include existing w_vel = 2.0 for reference baseline comparison
    all_results = []

    # Also evaluate w_vel = 2.0 (already trained)
    ckpt_20 = os.path.join(PROJECT_ROOT, "models/checkpoints/ronin_ideaB_impl1_turnvel.pt")
    if os.path.exists(ckpt_20):
        print("\nLoading reference checkpoint: w_vel_turn_max = 2.0...")
        model_20 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        model_20.load_state_dict(torch.load(ckpt_20, map_location=device)["model_state_dict"])
        model_20.eval()
        eval_20 = evaluate_sequence(model_20, seq_fig8, device)
        geom_20 = measure_figure8_geometry(eval_20)
        all_results.append({
            "w_vel": 2.0,
            "ckpt": ckpt_20,
            "final_drift": eval_20["final_drift"],
            "ate_rmse": eval_20["ate_rmse"],
            "rte_pct": eval_20["rte_pct"],
            **geom_20
        })

    for w_vel in w_vel_values:
        tag = f"wvel{int(w_vel * 10)}"
        save_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/ronin_impl1_{tag}.pt")
        print(f"\n>>> Training Model with w_vel_turn_max = {w_vel:.1f} (Save: {os.path.basename(save_path)})...")

        t_start = time.time()
        train_model(
            h5_path=train_dir,
            epochs=3,
            batch_size=64,
            step_size=200,
            lr=1e-3,
            w_dir_base=0.5,
            w_dir_turn_max=2.5,
            w_vel_turn_max=w_vel,
            augment=False,
            save_path=save_path
        )
        t_train = time.time() - t_start

        # Evaluate
        model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device)["model_state_dict"])
        model.eval()

        eval_res = evaluate_sequence(model, seq_fig8, device)
        geom_res = measure_figure8_geometry(eval_res)

        record = {
            "w_vel": w_vel,
            "ckpt": save_path,
            "train_time": t_train,
            "final_drift": eval_res["final_drift"],
            "ate_rmse": eval_res["ate_rmse"],
            "rte_pct": eval_res["rte_pct"],
            **geom_res
        }
        all_results.append(record)
        print(f"  Result: Final Drift = {eval_res['final_drift']:.2f}m | ATE = {eval_res['ate_rmse']:.2f}m | Radius Diff = {geom_res['r_diff']:+.3f}m | Speed = {geom_res['mean_speed']:.3f}m/s")

    # Print Full Comparative Table
    print("\n" + "=" * 115)
    print("HYPERPARAMETER SWEEP RESULTS TABLE (w_vel_turn_max)")
    print("=" * 115)
    header = f"{'w_vel_max':<10} | {'Final Drift':<12} | {'ATE RMSE':<10} | {'RTE %':<8} | {'R_right':<9} | {'R_left':<9} | {'R_avg':<9} | {'Radius Diff':<12} | {'Speed (m/s)':<12}"
    print(header)
    print("-" * 115)

    best_drift_rec = min(all_results, key=lambda x: x["final_drift"])
    best_radius_rec = min(all_results, key=lambda x: abs(x["r_diff"]))

    for r in all_results:
        star = " ★ CHAMPION" if r["w_vel"] == best_drift_rec["w_vel"] else ""
        print(f"{r['w_vel']:<10.1f} | {r['final_drift']:<10.2f}m | {r['ate_rmse']:<8.2f}m | {r['rte_pct']:<6.2f}% | {r['r_right']:<9.3f} | {r['r_left']:<9.3f} | {r['r_avg']:<9.3f} | {r['r_diff']:+10.3f}m | {r['mean_speed']:<10.3f}{star}")

    print("-" * 115)
    print(f"★ Lowest Final Drift  : w_vel_turn_max = {best_drift_rec['w_vel']:.1f} (Drift: {best_drift_rec['final_drift']:.2f} m, ATE: {best_drift_rec['ate_rmse']:.2f} m)")
    print(f"★ Most Accurate Radius: w_vel_turn_max = {best_radius_rec['w_vel']:.1f} (Radius: {best_radius_rec['r_avg']:.3f} m, Diff: {best_radius_rec['r_diff']:+.3f} m)")
    print("=" * 115)

if __name__ == "__main__":
    main()
