"""
Testing User's Solution:
Apply the 1.2s Transient Fallback Filter to the New Retrained Model (model_fb_120.pt)
to fix Pocket Loop and Handheld Walk!
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector
from scripts.sweep_fallback_time import preextract_sequence, run_inference, score_trajectory, apply_dwell_filter

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")

    # Load the retrained model (model_fb_120.pt) which gave Bag=5.88m and Fig8=12.67m
    model_path = os.path.join(PROJECT_ROOT, "models/checkpoints/fallback_models/model_fb_120.pt")
    m_new = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m_new.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    m_new.eval()

    # Also load the stable straight reference model (m21) for fallback
    m21 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m21.load_state_dict(torch.load("models/checkpoints/ronin_impl1_wvel21.pt", map_location=device)["model_state_dict"])
    m21.eval()

    holdouts = [
        ("Pocket Loop", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "pocket"),
        ("Handheld Walk", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "handheld"),
        ("Bag Corridor", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "bag"),
        ("Figure-8 Track", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "pocket"),
    ]

    print("Evaluating User's Filter Solution on the Retrained Model...")
    threshold = 0.23
    dwell_sec = 1.2

    for desc, path, pose in holdouts:
        fpath = os.path.join(PROJECT_ROOT, path)
        seq = PreprocessedSequence(fpath, beta=0.15)
        ext = preextract_sequence(seq, step_size=20, window_size=200)

        # 1. Standalone raw output (what gave 41.97m, 86.82m, 5.88m, 12.67m)
        v_raw = run_inference(m_new, ext["pred_windows"], device)
        d_raw, rte_raw, ate_raw = score_trajectory(v_raw, ext["anchor_yaws"], ext["pos_gt"], ext["dt_step"])

        # 2. Applying User's 1.2s Transient Fallback Filter:
        # When a turn is quick (< 1.2s like 90° corners or wrist twitch), fallback to stable 2.1 straight mode!
        # When a turn is sustained (>= 1.2s), use the new model that mastered Figure-8!
        v_21 = run_inference(m21, ext["pred_windows"], device)
        dt_step = ext["dt_step"]
        dwell_steps = int(round(dwell_sec / dt_step))
        curve_mask = apply_dwell_filter(ext["turn_rates"], threshold, dwell_steps)

        v_filtered = np.where(curve_mask[:, None], v_raw, v_21)
        d_filt, rte_filt, ate_filt = score_trajectory(v_filtered, ext["anchor_yaws"], ext["pos_gt"], dt_step)

        print(f"\n{desc} [{pose.upper()}]:")
        print(f"  • Standalone Raw (Unfiltered)    : Drift = {d_raw:>6.2f}m ({rte_raw:.2f}%) | ATE = {ate_raw:>5.2f}m")
        print(f"  • WITH USER'S 1.2s FILTER SOLUTION: Drift = {d_filt:>6.2f}m ({rte_filt:.2f}%) | ATE = {ate_filt:>5.2f}m")

if __name__ == "__main__":
    main()
