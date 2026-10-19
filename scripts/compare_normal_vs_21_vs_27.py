"""
Comprehensive Comparison: Normal 1 (w_vel=1.0) vs 2.1 vs 2.7 across All Real-World Holdout Walks
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence
from scripts.compare_journey_errors import get_journey_errors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        ("Normal 1 (Baseline w_vel=1.0, w_dir=0.5)", "models/checkpoints/ronin_resnet_best.pt"),
        ("Idea B (Normal w_vel=1.0, w_dir=2.5)", "models/checkpoints/ronin_ideaB_max25.pt"),
        ("Tuned 2.1 (w_vel=2.1, w_dir=2.5)", "models/checkpoints/ronin_impl1_wvel21.pt"),
        ("Tuned 2.7 (w_vel=2.7, w_dir=2.5)", "models/checkpoints/ronin_impl1_wvel27.pt"),
    ]

    test_sequences = [
        ("1. Pocket Loop (test_sub01)", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "Real World / Daily Life"),
        ("2. Handheld Walk (test_sub02)", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "Real World / Daily Life"),
        ("3. Bag Corridor (test_sub03)", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "Real World / Daily Life"),
        ("4. Curved Figure-8 (test_sub04)", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "Curved Path / Figure-8"),
    ]

    print("=" * 115)
    print("REAL-WORLD COMPARISON: NORMAL 1.0 vs 2.1 vs 2.7")
    print("=" * 115)

    for seq_name, seq_path, cat in test_sequences:
        full_seq_path = os.path.join(PROJECT_ROOT, seq_path)
        seq = PreprocessedSequence(full_seq_path, beta=0.15)
        print(f"\n>>> {seq_name.upper()} [{cat}] - Total Distance: {seq.pos_gt.shape[0]*seq.dt*1.1:.0f}m")
        print("-" * 115)
        print(f"{'Model Configuration':<42} | {'Final Drift':<16} | {'ATE RMSE (Journey)':<20} | {'RTE %':<10}")
        print("-" * 115)

        for m_name, m_path in models:
            full_m_path = os.path.join(PROJECT_ROOT, m_path)
            model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
            model.load_state_dict(torch.load(full_m_path, map_location=device)["model_state_dict"])
            model.eval()

            res = get_journey_errors(model, seq, device)
            traj_dist = float(np.sum(np.linalg.norm(np.diff(seq.pos_gt[::10, :2], axis=0), axis=1)))
            rte_pct = (res["final_error"] / traj_dist) * 100.0 if traj_dist > 0 else 0.0

            print(f"{m_name:<42} | {res['final_error']:>7.2f} m        | {res['ate_rmse']:>7.2f} m             | {rte_pct:>6.2f}%")

    print("\n" + "=" * 115)

if __name__ == "__main__":
    main()
