"""
Journey-Wide Trajectory Error Comparison
Compares w_vel = 2.0 vs 2.1 vs 2.7 across the entire 44-minute walk, not just the final endpoint.
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame

def get_journey_errors(model, seq, device, step_size=10, window_size=200):
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

    time_min = np.arange(len(errors)) * dt_step / 60.0

    return {
        "errors": errors,
        "time_min": time_min,
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "p90_error": float(np.percentile(errors, 90)),
        "max_error": float(np.max(errors)),
        "final_error": float(errors[-1]),
        "err_10m": float(errors[np.argmin(np.abs(time_min - 10.0))]),
        "err_20m": float(errors[np.argmin(np.abs(time_min - 20.0))]),
        "err_30m": float(errors[np.argmin(np.abs(time_min - 30.0))]),
        "err_40m": float(errors[np.argmin(np.abs(time_min - 40.0))]),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [
        ("w_vel = 2.0", "models/checkpoints/ronin_ideaB_impl1_turnvel.pt"),
        ("w_vel = 2.1", "models/checkpoints/ronin_impl1_wvel21.pt"),
        ("w_vel = 2.7", "models/checkpoints/ronin_impl1_wvel27.pt"),
    ]

    fig8_path = os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5")
    seq_fig8 = PreprocessedSequence(fig8_path, beta=0.15)

    print("=" * 95)
    print("JOURNEY-WIDE ERROR ANALYSIS ON 44-MINUTE FIGURE-8 WALK")
    print("=" * 95)

    fig8_results = {}
    for name, rel_path in candidates:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        m.load_state_dict(torch.load(full_path, map_location=device)["model_state_dict"])
        m.eval()
        res = get_journey_errors(m, seq_fig8, device)
        fig8_results[name] = res

    # Journey metrics table
    print(f"{'Metric':<32} | {'w_vel = 2.0':<18} | {'w_vel = 2.1':<18} | {'w_vel = 2.7':<18}")
    print("-" * 95)
    print(f"{'ATE RMSE (Overall Journey)':<32} | {fig8_results['w_vel = 2.0']['ate_rmse']:<18.2f}m | {fig8_results['w_vel = 2.1']['ate_rmse']:<18.2f}m | {fig8_results['w_vel = 2.7']['ate_rmse']:<18.2f}m")
    print(f"{'Mean Error (Avg across walk)':<32} | {fig8_results['w_vel = 2.0']['mean_error']:<18.2f}m | {fig8_results['w_vel = 2.1']['mean_error']:<18.2f}m | {fig8_results['w_vel = 2.7']['mean_error']:<18.2f}m")
    print(f"{'Median Error (50% of walk)':<32} | {fig8_results['w_vel = 2.0']['median_error']:<18.2f}m | {fig8_results['w_vel = 2.1']['median_error']:<18.2f}m | {fig8_results['w_vel = 2.7']['median_error']:<18.2f}m")
    print(f"{'90th Percentile Error':<32} | {fig8_results['w_vel = 2.0']['p90_error']:<18.2f}m | {fig8_results['w_vel = 2.1']['p90_error']:<18.2f}m | {fig8_results['w_vel = 2.7']['p90_error']:<18.2f}m")
    print(f"{'Max Peak Drift During Walk':<32} | {fig8_results['w_vel = 2.0']['max_error']:<18.2f}m | {fig8_results['w_vel = 2.1']['max_error']:<18.2f}m | {fig8_results['w_vel = 2.7']['max_error']:<18.2f}m")
    print("-" * 95)
    print("PROGRESSION MILESTONES ACROSS THE 44 MINUTES:")
    print(f"{'• Drift at Minute 10 (~1/4 walk)':<32} | {fig8_results['w_vel = 2.0']['err_10m']:<18.2f}m | {fig8_results['w_vel = 2.1']['err_10m']:<18.2f}m | {fig8_results['w_vel = 2.7']['err_10m']:<18.2f}m")
    print(f"{'• Drift at Minute 20 (~1/2 walk)':<32} | {fig8_results['w_vel = 2.0']['err_20m']:<18.2f}m | {fig8_results['w_vel = 2.1']['err_20m']:<18.2f}m | {fig8_results['w_vel = 2.7']['err_20m']:<18.2f}m")
    print(f"{'• Drift at Minute 30 (~3/4 walk)':<32} | {fig8_results['w_vel = 2.0']['err_30m']:<18.2f}m | {fig8_results['w_vel = 2.1']['err_30m']:<18.2f}m | {fig8_results['w_vel = 2.7']['err_30m']:<18.2f}m")
    print(f"{'• Drift at Minute 40 (Near end)':<32} | {fig8_results['w_vel = 2.0']['err_40m']:<18.2f}m | {fig8_results['w_vel = 2.1']['err_40m']:<18.2f}m | {fig8_results['w_vel = 2.7']['err_40m']:<18.2f}m")
    print(f"{'• Final Drift at Finish (44 min)':<32} | {fig8_results['w_vel = 2.0']['final_error']:<18.2f}m | {fig8_results['w_vel = 2.1']['final_error']:<18.2f}m | {fig8_results['w_vel = 2.7']['final_error']:<18.2f}m")
    print("-" * 95)

    # Journey Error on other unseen holdout test sequences
    print("\nJOURNEY-WIDE ATE RMSE ACROSS ALL OTHER UNSEEN WALKS:")
    test_files = [
        ("Pocket Loop (test_sub01)", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5"),
        ("Handheld Mixed (test_sub02)", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5"),
        ("Bag Corridor (test_sub03)", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5"),
    ]
    for sname, spath in test_files:
        full_spath = os.path.join(PROJECT_ROOT, spath)
        seq = PreprocessedSequence(full_spath, beta=0.15)
        row = []
        for name, rel_path in candidates:
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
            m.load_state_dict(torch.load(full_path, map_location=device)["model_state_dict"])
            m.eval()
            res = get_journey_errors(m, seq, device)
            row.append(f"{res['ate_rmse']:.2f}m")
        print(f"{sname:<32} | {row[0]:<18} | {row[1]:<18} | {row[2]:<18}")
    print("=" * 95)

if __name__ == "__main__":
    main()
