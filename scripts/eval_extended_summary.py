"""
Evaluates w_vel = 2.6, 2.7, 2.8, 2.9, 3.0 and prints the comparison table
"""
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence
from scripts.sweep_w_vel import evaluate_sequence, measure_figure8_geometry

def main():
    seq = PreprocessedSequence("data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", beta=0.15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        (2.0, "models/checkpoints/ronin_ideaB_impl1_turnvel.pt"),
        (2.5, "models/checkpoints/ronin_impl1_wvel25.pt"),
        (2.6, "models/checkpoints/ronin_impl1_wvel26.pt"),
        (2.7, "models/checkpoints/ronin_impl1_wvel27.pt"),
        (2.8, "models/checkpoints/ronin_impl1_wvel28.pt"),
        (2.9, "models/checkpoints/ronin_impl1_wvel29.pt"),
        (3.0, "models/checkpoints/ronin_impl1_wvel30.pt"),
    ]

    print("=" * 90)
    print("EXTENDED SWEEP EVALUATION (w_vel_turn_max from 2.0 to 3.0)")
    print("=" * 90)
    header = f"{'w_vel':<8} | {'Final Drift':<12} | {'ATE RMSE':<10} | {'RTE %':<8} | {'R_avg':<9} | {'Radius Diff':<12} | {'Speed':<8}"
    print(header)
    print("-" * 90)

    for w, rel_path in models:
        path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(path):
            continue
        m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
        m.eval()
        ev = evaluate_sequence(m, seq, device)
        geom = measure_figure8_geometry(ev)
        print(f"{w:<8.1f} | {ev['final_drift']:>9.2f} m | {ev['ate_rmse']:>7.2f} m | {ev['rte_pct']:>5.2f}% | {geom['r_avg']:>6.3f}m | {geom['r_diff']:>+10.3f}m | {geom['mean_speed']:>6.3f}m/s")
    print("=" * 90)

if __name__ == "__main__":
    main()
