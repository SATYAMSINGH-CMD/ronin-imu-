"""
Check progress of dual regime models completed so far
"""
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence
from scripts.sweep_w_vel import evaluate_sequence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_fig8 = PreprocessedSequence(os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5"), beta=0.15)
    seq_pocket = PreprocessedSequence(os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5"), beta=0.15)

    print("=" * 80)
    print("EARLY RESULTS FROM DUAL-REGIME THRESHOLD SWEEP")
    print("=" * 80)
    for thresh in [15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]:
        p = os.path.join(PROJECT_ROOT, f"models/checkpoints/ronin_dual_regime_thresh_{thresh}.pt")
        if os.path.exists(p):
            m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
            m.load_state_dict(torch.load(p, map_location=device)["model_state_dict"])
            m.eval()
            res_f = evaluate_sequence(m, seq_fig8, device)
            res_p = evaluate_sequence(m, seq_pocket, device)
            print(f"Threshold {thresh/100:.2f} rad/s ({thresh/100*180/3.14159:.1f}°/s): Fig-8 Drift = {res_f['final_drift']:>6.2f}m ({res_f['rte_pct']:.2f}%) | Pocket Drift = {res_p['final_drift']:>6.2f}m ({res_p['rte_pct']:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
