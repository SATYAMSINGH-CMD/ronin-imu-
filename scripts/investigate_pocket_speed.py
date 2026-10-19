import os
import sys
import torch
import numpy as np

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector
from scripts.sweep_fallback_time import preextract_sequence, run_inference, apply_dwell_filter

def main():
    device = torch.device("cpu")
    print("=" * 115, flush=True)
    print("POCKET vs HANDHELD vs BAG SPEED & DISTANCE INVESTIGATION (SOLUTION A 1.1s FALLBACK)", flush=True)
    print("=" * 115, flush=True)

    # 1. Load models
    p21 = os.path.join(PROJECT_ROOT, "models/checkpoints/real_v3/model_straight_wvel21.pt")
    p27 = os.path.join(PROJECT_ROOT, "models/checkpoints/real_v3/model_curve_wvel27.pt")

    m21 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m21.load_state_dict(torch.load(p21, map_location=device)["model_state_dict"])
    m21.eval()

    m27 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m27.load_state_dict(torch.load(p27, map_location=device)["model_state_dict"])
    m27.eval()

    sequences = [
        ("Pocket Corridor Loop", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "POCKET"),
        ("44-Min Figure-8 Loop", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "POCKET"),
        ("Natural Handheld Walk", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "HANDHELD"),
        ("Bag Corridor Walk", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "BAG"),
    ]

    for name, rel_path, pose in sequences:
        print(f"\nProcessing {name} [{pose}]...", flush=True)
        fpath = os.path.join(PROJECT_ROOT, rel_path)
        seq = PreprocessedSequence(fpath, beta=0.15)
        ext = preextract_sequence(seq, step_size=20, window_size=200)

        dt_step = ext["dt_step"]
        num_wins = ext["num_wins"]
        pos_gt = ext["pos_gt"]

        v21_pred = run_inference(m21, ext["pred_windows"], device)
        v27_pred = run_inference(m27, ext["pred_windows"], device)

        dwell_steps = int(round(1.10 / dt_step))
        mask_solA = apply_dwell_filter(ext["turn_rates"], 0.23, dwell_steps)
        v_solA = np.where(mask_solA[:, None], v27_pred, v21_pred)

        # Integrate trajectory
        pos_solA = np.zeros((num_wins, 2), dtype=np.float32)
        pos_solA[0] = pos_gt[0]
        for k in range(1, num_wins):
            v_w = rotate_2d_vector(v_solA[k], +ext["anchor_yaws"][k])
            pos_solA[k] = pos_solA[k-1] + v_w * dt_step

        # True path distance
        step_dists_gt = np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)
        tot_dist_gt = float(np.sum(step_dists_gt))

        # Predicted path distance
        speed_solA = np.linalg.norm(v_solA, axis=1)
        tot_dist_pred = float(np.sum(speed_solA[:-1] * dt_step))

        dist_diff = tot_dist_pred - tot_dist_gt
        scale_ratio = tot_dist_pred / tot_dist_gt if tot_dist_gt > 0 else 1.0

        avg_speed_gt = tot_dist_gt / (num_wins * dt_step)
        avg_speed_pred = tot_dist_pred / (num_wins * dt_step)

        # Final endpoint drift
        final_drift = float(np.linalg.norm(pos_solA[-1] - pos_gt[-1]))
        rte_pct = (final_drift / tot_dist_gt) * 100.0 if tot_dist_gt > 0 else 0.0

        # Continuous error stats
        ate_errors = np.linalg.norm(pos_solA - pos_gt, axis=1)
        ate_rmse = float(np.sqrt(np.mean(ate_errors ** 2)))
        ate_max = float(np.max(ate_errors))

        # Vector decomposition
        vec_err = pos_solA[-1] - pos_gt[-1]
        vec_gt = pos_gt[-1] - pos_gt[0]
        disp_gt = float(np.linalg.norm(vec_gt))

        print(f"==========================================================================", flush=True)
        print(f"RESULTS FOR: {name.upper()} [{pose}]", flush=True)
        print(f"==========================================================================", flush=True)
        print(f"  • Ground Truth Total Distance Walked: {tot_dist_gt:>8.2f} m  (Avg Speed: {avg_speed_gt:.3f} m/s)", flush=True)
        print(f"  • Predicted Distance Walked (Sol A):  {tot_dist_pred:>8.2f} m  (Avg Speed: {avg_speed_pred:.3f} m/s)", flush=True)
        print(f"  • Distance Discrepancy (Pred - True): {dist_diff:>+8.2f} m", flush=True)
        print(f"  • Velocity Scale Ratio:               {scale_ratio*100:>8.2f}%", flush=True)
        print(f"  • Final Endpoint Drift:               {final_drift:>8.2f} m  ({rte_pct:.2f}% RTE)", flush=True)
        print(f"  • Continuous ATE Error (RMSE / Max):  {ate_rmse:>8.2f} m / {ate_max:>6.2f} m", flush=True)
        
        if disp_gt > 5.0:
            u_gt = vec_gt / disp_gt
            along_err = float(np.dot(vec_err, u_gt))
            cross_err = float(np.linalg.norm(vec_err - along_err * u_gt))
            print(f"  • Endpoint Error Breakdown:           Along-Track: {along_err:+6.2f}m | Cross-Track: {cross_err:6.2f}m", flush=True)
        else:
            print(f"  • Endpoint Error Breakdown:           Closed Loop (Net displacement: {disp_gt:.2f}m)", flush=True)
        print(f"==========================================================================\n", flush=True)

if __name__ == "__main__":
    main()
