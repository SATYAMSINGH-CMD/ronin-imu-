"""
Extended Hyperparameter Sweep for w_vel_turn_max: [2.7, 2.8, 2.9, 3.0]
Tests if increasing w_vel beyond 2.6 decreases drift further or oscillates/diverges.
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
from src.data.dataset import PreprocessedSequence
from scripts.sweep_w_vel import evaluate_sequence, measure_figure8_geometry

def main():
    print("=" * 105)
    print("EXTENDED SWEEP: w_vel_turn_max in [2.7, 2.8, 2.9, 3.0]")
    print("=" * 105)

    w_vel_values = [2.7, 2.8, 2.9, 3.0]
    train_dir = os.path.join(PROJECT_ROOT, "data/real/train")
    fig8_path = os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_fig8 = PreprocessedSequence(fig8_path, beta=0.15)
    results = []

    for w_vel in w_vel_values:
        tag = f"wvel{int(w_vel * 10)}"
        save_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/ronin_impl1_{tag}.pt")
        print(f"\n>>> Training with w_vel_turn_max = {w_vel:.1f}...")

        t0 = time.time()
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
        elapsed = time.time() - t0

        model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device)["model_state_dict"])
        model.eval()

        eval_res = evaluate_sequence(model, seq_fig8, device)
        geom = measure_figure8_geometry(eval_res)

        rec = {
            "w_vel": w_vel,
            "final_drift": eval_res["final_drift"],
            "ate_rmse": eval_res["ate_rmse"],
            "rte_pct": eval_res["rte_pct"],
            "time": elapsed,
            **geom
        }
        results.append(rec)
        print(f"  Result (w_vel={w_vel}): Drift = {eval_res['final_drift']:.2f}m | ATE = {eval_res['ate_rmse']:.2f}m | Radius Diff = {geom['r_diff']:+.3f}m | Speed = {geom['mean_speed']:.3f}m/s")

    print("\n" + "=" * 115)
    print("EXTENDED SWEEP RESULTS TABLE")
    print("=" * 115)
    header = f"{'w_vel_max':<10} | {'Final Drift':<12} | {'ATE RMSE':<10} | {'RTE %':<8} | {'R_avg':<9} | {'Radius Diff':<12} | {'Speed (m/s)':<12}"
    print(header)
    print("-" * 115)
    for r in results:
        print(f"{r['w_vel']:<10.1f} | {r['final_drift']:<10.2f}m | {r['ate_rmse']:<8.2f}m | {r['rte_pct']:<6.2f}% | {r['r_avg']:<9.3f} | {r['r_diff']:+10.3f}m | {r['mean_speed']:<10.3f}")
    print("=" * 115)

if __name__ == "__main__":
    main()
