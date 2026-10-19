"""
Precise Geometric & Kinematic Radius Measurement for Figure-8 Trajectory
Compares Ground Truth vs Baseline vs Idea B vs Implementation 1
"""

import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame

def fit_circle_least_squares(x, y):
    """
    Fits a 2D circle (xc, yc, R) to points (x, y) minimizing sum((sqrt((x-xc)^2 + (y-yc)^2) - R)^2).
    Uses algebraic initial guess via Kasa/Pratt method then refines.
    """
    # Algebraic initial guess
    # x^2 + y^2 = 2*x*xc + 2*y*yc + (R^2 - xc^2 - yc^2)
    # A * [2*xc, 2*yc, C].T = x^2 + y^2
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc0, yc0, C = sol[0], sol[1], sol[2]
    R0 = np.sqrt(max(1e-4, xc0**2 + yc0**2 + C))

    def loss(params):
        xc, yc, R = params
        dist = np.sqrt((x - xc)**2 + (y - yc)**2)
        return np.sum((dist - R)**2)

    res = minimize(loss, [xc0, yc0, R0], method="Nelder-Mead")
    xc, yc, R = res.x
    # Residual error
    residuals = np.abs(np.sqrt((x - xc)**2 + (y - yc)**2) - R)
    rmse = np.sqrt(np.mean(residuals**2))
    return xc, yc, R, rmse

def run_inference(model, seq, device, window_size=200, step_size=10):
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
    return pos_pred, pos_gt, v_world_preds, dt_step

def analyze_models():
    h5_path = "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5"
    seq = PreprocessedSequence(h5_path, beta=0.15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = {
        "Ground Truth": None,
        "Baseline (w_dir=0.5 static)": "models/checkpoints/ronin_resnet_best.pt",
        "Idea B (w_dir=2.5 adaptive, w_vel=1.0)": "models/checkpoints/ronin_ideaB_max25.pt",
        "Implementation 1 (w_dir=2.5, w_vel=2.0)": "models/checkpoints/ronin_ideaB_impl1_turnvel.pt",
    }

    results = {}

    for name, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            # Ground truth
            pass
        else:
            model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            pos_pred, pos_gt, v_world, dt_step = run_inference(model, seq, device)
            results[name] = {
                "pos": pos_pred,
                "v_world": v_world,
                "dt_step": dt_step
            }

    # Also ground truth
    # 1 loop = 52.4 seconds -> 52.4 / 0.05 = ~1048 steps
    # Right circle is first half (0 to 26.2s -> 524 steps)
    # Left circle is second half (26.2s to 52.4s -> steps 524 to 1048)
    dt_step = 0.05
    steps_half_loop = int(26.2 / dt_step)
    steps_full_loop = int(52.4 / dt_step)

    print("=" * 80)
    print(f"{'Model':<40} | {'R_right (m)':<12} | {'R_left (m)':<12} | {'R_avg (m)':<10} | {'Diff vs 5.0m':<12} | {'Shrinkage %':<10}")
    print("-" * 80)

    # Let's measure for Loop 1 (first 52.4 seconds)
    # For Ground Truth
    gt_pos = np.array([seq.pos_gt[i * 10, :2] for i in range(len(results["Baseline (w_dir=0.5 static)"]["pos"]))])
    
    all_trajs = {"Ground Truth": gt_pos}
    for k, v in results.items():
        all_trajs[k] = v["pos"]

    # Measure across Loop 1 (steps 0 to 1048)
    print("\n--- LOOP 1 (Minutes 0.0 to 0.87, First Complete 8-Figure) ---")
    for name, pos in all_trajs.items():
        # Right lobe: steps 0 to 524
        x_r, y_r = pos[:steps_half_loop, 0], pos[:steps_half_loop, 1]
        _, _, r_right, _ = fit_circle_least_squares(x_r, y_r)

        # Left lobe: steps 524 to 1048
        x_l, y_l = pos[steps_half_loop:steps_full_loop, 0], pos[steps_half_loop:steps_full_loop, 1]
        _, _, r_left, _ = fit_circle_least_squares(x_l, y_l)

        r_avg = (r_right + r_left) / 2.0
        diff = r_avg - 5.0
        shrink_pct = ((r_avg - 5.0) / 5.0) * 100.0

        print(f"{name:<40} | {r_right:<12.3f} | {r_left:<12.3f} | {r_avg:<10.3f} | {diff:+11.3f}m | {shrink_pct:+9.2f}%")

    # Measure across Loops 1 to 3 (Average of first 3 loops)
    print("\n--- LOOPS 1 to 3 (Average of First 3 Complete 8-Figures) ---")
    for name, pos in all_trajs.items():
        r_rights = []
        r_lefts = []
        for loop_idx in range(3):
            s_r = loop_idx * steps_full_loop
            e_r = s_r + steps_half_loop
            s_l = e_r
            e_l = s_r + steps_full_loop
            _, _, rr, _ = fit_circle_least_squares(pos[s_r:e_r, 0], pos[s_r:e_r, 1])
            _, _, rl, _ = fit_circle_least_squares(pos[s_l:e_l, 0], pos[s_l:e_l, 1])
            r_rights.append(rr)
            r_lefts.append(rl)
        
        mean_rr = np.mean(r_rights)
        mean_rl = np.mean(r_lefts)
        mean_r = (mean_rr + mean_rl) / 2.0
        diff = mean_r - 5.0
        shrink_pct = ((mean_r - 5.0) / 5.0) * 100.0
        print(f"{name:<40} | {mean_rr:<12.3f} | {mean_rl:<12.3f} | {mean_r:<10.3f} | {diff:+11.3f}m | {shrink_pct:+9.2f}%")

    # Kinematic check: Speed during curves
    print("\n--- KINEMATIC CHECK: Mean Predicted Walking Speed on Turns (True = 1.150 m/s) ---")
    for name, v_dict in results.items():
        v = v_dict["v_world"]
        speed = np.linalg.norm(v, axis=1)
        mean_speed = np.mean(speed)
        speed_diff = mean_speed - 1.15
        speed_ratio = (mean_speed / 1.15) * 100.0
        print(f"{name:<40} | Mean Speed: {mean_speed:.3f} m/s | Diff: {speed_diff:+.3f} m/s | Stride Length Ratio: {speed_ratio:.1f}%")

if __name__ == "__main__":
    analyze_models()
