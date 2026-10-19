"""
Exact Figure-8 Loop Measurement Script
Evaluates Ground Truth vs Implementation 1 across individual Figure-8 cycles.
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.measure_radius import run_inference, fit_circle_least_squares
from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence

def main():
    seq = PreprocessedSequence("data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", beta=0.15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("models/checkpoints/ronin_ideaB_impl1_turnvel.pt", map_location=device)
    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pos_pred, pos_gt, v_world, dt_step = run_inference(model, seq, device)

    # Exact period:
    R_true = 5.0
    circ = 2.0 * np.pi * R_true
    T_circle = circ / 1.15  # 27.31817 s
    T_fig8 = 2.0 * T_circle  # 54.63634 s

    steps_circle = int(round(T_circle / dt_step))  # 546 steps
    steps_fig8 = 2 * steps_circle                  # 1092 steps

    print("=" * 95)
    print("EXACT INDIVIDUAL LOOP MEASUREMENTS (Implementation 1 vs Ground Truth)")
    print(f"True Cycle Period: {T_fig8:.2f}s ({steps_fig8} steps) | Circle Period: {T_circle:.2f}s ({steps_circle} steps)")
    print("-" * 95)
    header = f"{'Loop':<8} | {'True R_right':<13} | {'True R_left':<12} | {'Pred R_right':<13} | {'Pred R_left':<12} | {'R_avg Diff':<12} | {'Shrinkage %':<12}"
    print(header)
    print("-" * 95)

    diffs = []
    shrinks = []
    for i in range(10):
        s = i * steps_fig8
        mid = s + steps_circle
        e = (i + 1) * steps_fig8
        if e > len(pos_pred):
            break

        # Ground truth
        _, _, r_r_gt, _ = fit_circle_least_squares(pos_gt[s:mid, 0], pos_gt[s:mid, 1])
        _, _, r_l_gt, _ = fit_circle_least_squares(pos_gt[mid:e, 0], pos_gt[mid:e, 1])

        # Prediction
        _, _, r_r_pr, _ = fit_circle_least_squares(pos_pred[s:mid, 0], pos_pred[s:mid, 1])
        _, _, r_l_pr, _ = fit_circle_least_squares(pos_pred[mid:e, 0], pos_pred[mid:e, 1])

        r_gt_avg = (r_r_gt + r_l_gt) / 2.0
        r_pr_avg = (r_r_pr + r_l_pr) / 2.0
        diff = r_pr_avg - r_gt_avg
        shrink = (diff / r_gt_avg) * 100.0
        diffs.append(diff)
        shrinks.append(shrink)

        print(f"Loop {i+1:<3} | {r_r_gt:<13.3f} | {r_l_gt:<12.3f} | {r_r_pr:<13.3f} | {r_l_pr:<12.3f} | {diff:+11.3f}m | {shrink:+10.2f}%")

    print("-" * 95)
    print(f"Average Radius Difference across Loops: {np.mean(diffs):+.3f} meters ({np.mean(shrinks):+.2f}%)")
    print(f"Mean Predicted Speed: {np.linalg.norm(v_world, axis=1).mean():.3f} m/s (True: 1.150 m/s)")
    print("=" * 95)

if __name__ == "__main__":
    main()
