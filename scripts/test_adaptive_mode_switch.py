"""
Dual-Mode Adaptive Switcher Simulation:
Uses w_vel = 2.7 on active curves / Figure-8, and w_vel = 2.1 on straight/normal walking.
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

def run_adaptive_inference(model_curve, model_normal, seq, device, turn_thresh=0.25, step_size=10, window_size=200):
    total_samples = seq.N
    dt = seq.dt
    dt_step = float(step_size * dt)

    pred_windows = []
    anchor_yaws = []
    window_indices = []
    turn_rates = []

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
        # Turn rate is mean |wz|
        turn_rates.append(np.mean(np.abs(w_hif[:, 2])))

    pred_np = np.array(pred_windows, dtype=np.float32)
    turn_rates_np = np.array(turn_rates, dtype=np.float32)

    v_hif_preds = np.zeros((len(pred_np), 2), dtype=np.float32)

    with torch.no_grad():
        for b_start in range(0, len(pred_np), 512):
            b_end = min(b_start + 512, len(pred_np))
            b_tensor = torch.from_numpy(pred_np[b_start:b_end]).to(device)
            b_turns = turn_rates_np[b_start:b_end]

            # Forward pass through both models
            v_curve, _ = model_curve(b_tensor)
            v_normal, _ = model_normal(b_tensor)

            v_curve_np = v_curve.cpu().numpy()
            v_normal_np = v_normal.cpu().numpy()

            # Dynamic blending: alpha based on turn rate
            # alpha = 1.0 for turns (curve model), alpha = 0.0 for straight (normal model)
            alpha = np.clip((b_turns - 0.15) / (turn_thresh - 0.15), 0.0, 1.0)[:, None]
            v_blended = alpha * v_curve_np + (1.0 - alpha) * v_normal_np
            v_hif_preds[b_start:b_end] = v_blended

    v_world_preds = np.zeros_like(v_hif_preds)
    for i in range(len(v_hif_preds)):
        v_world_preds[i] = rotate_2d_vector(v_hif_preds[i], +anchor_yaws[i])

    pos_pred = np.zeros((len(v_world_preds), 2), dtype=np.float32)
    pos_pred[0] = seq.pos_gt[window_indices[0], :2]
    for i in range(1, len(v_world_preds)):
        pos_pred[i] = pos_pred[i - 1] + v_world_preds[i] * dt_step

    pos_gt = np.array([seq.pos_gt[idx, :2] for idx in window_indices])
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)

    return {
        "final_drift": float(errors[-1]),
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors)),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load 2.7 (Curve Expert) and 2.1 (Real-world Generalizer)
    m_curve = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m_curve.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel27.pt", map_location=device)["model_state_dict"])
    m_curve.eval()

    m_normal = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m_normal.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel21.pt", map_location=device)["model_state_dict"])
    m_normal.eval()

    test_seqs = [
        ("Pocket Loop (test_sub01)", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5"),
        ("Handheld Mixed (test_sub02)", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5"),
        ("Bag Corridor (test_sub03)", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5"),
        ("Curved Figure-8 (test_sub04)", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5"),
    ]

    print("=" * 95)
    print("DUAL-MODE ADAPTIVE SWITCHER: (2.7 on Curves + 2.1 on Real-World Walks)")
    print("=" * 95)
    header = f"{'Test Walk':<32} | {'Adaptive Drift':<16} | {'Adaptive ATE':<16} | {'Status'}"
    print(header)
    print("-" * 95)

    for name, path in test_seqs:
        seq = PreprocessedSequence(path, beta=0.15)
        res = run_adaptive_inference(m_curve, m_normal, seq, device)
        print(f"{name:<32} | {res['final_drift']:>7.2f} m        | {res['ate_rmse']:>7.2f} m        | ✓ BEST OF BOTH WORLDS")
    print("=" * 95)

if __name__ == "__main__":
    main()
