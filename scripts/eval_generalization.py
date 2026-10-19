"""
Evaluation of generalization across other unseen test sequences
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
from scripts.sweep_w_vel import evaluate_sequence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = {
        "w_vel=2.0": "models/checkpoints/ronin_ideaB_impl1_turnvel.pt",
        "w_vel=2.1": "models/checkpoints/ronin_impl1_wvel21.pt",
        "w_vel=2.7": "models/checkpoints/ronin_impl1_wvel27.pt",
        "w_vel=2.8": "models/checkpoints/ronin_impl1_wvel28.pt",
    }

    test_files = [
        ("Pocket Loop (test_sub01)", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5"),
        ("Handheld Mixed (test_sub02)", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5"),
        ("Bag Corridor (test_sub03)", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5"),
    ]

    print("=" * 105)
    print("GENERALIZATION CHECK ON OTHER UNSEEN TEST SEQUENCES")
    print("=" * 105)
    header = f"{'Sequence':<30} | {'w_vel=2.0 Drift':<16} | {'w_vel=2.1 Drift':<16} | {'w_vel=2.2 Drift':<16} | {'w_vel=2.6 Drift':<16}"
    print(header)
    print("-" * 105)

    for name, path in test_files:
        full_path = os.path.join(PROJECT_ROOT, path)
        seq = PreprocessedSequence(full_path, beta=0.15)
        drifts = []
        for cname, cpath in ckpts.items():
            full_cpath = os.path.join(PROJECT_ROOT, cpath)
            m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
            m.load_state_dict(torch.load(full_cpath, map_location=device)["model_state_dict"])
            m.eval()
            res = evaluate_sequence(m, seq, device)
            drifts.append(f"{res['final_drift']:.2f}m ({res['rte_pct']:.2f}%)")
        print(f"{name:<30} | {drifts[0]:<16} | {drifts[1]:<16} | {drifts[2]:<16} | {drifts[3]:<16}")
    print("=" * 105)

if __name__ == "__main__":
    main()
