import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence
from scripts.sweep_w_vel import evaluate_sequence
from scripts.generate_and_eval_500mb import DIVERSE_TEST_SPECS

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")
    
    model_path = os.path.join(PROJECT_ROOT, "models/checkpoints/ronin_resnet_best.pt")
    m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    m.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    m.eval()

    div_dir = os.path.join(PROJECT_ROOT, "data/real/test_diverse_500mb")
    print("Evaluating Sachini (RoNIN Baseline) on 10 Diverse Sequences...", flush=True)
    
    total_drift = 0.0
    total_ate = 0.0
    
    for spec in DIVERSE_TEST_SPECS:
        fpath = os.path.join(div_dir, f"{spec['id']}.h5")
        seq = PreprocessedSequence(fpath, beta=0.15)
        res = evaluate_sequence(m, seq, device, step_size=20)
        total_drift += res["final_drift"]
        total_ate += res["ate_rmse"]
        print(f"  {spec['desc']:<30} [{spec['pose'].upper():<8}]: Drift = {res['final_drift']:>6.2f}m ({res['rte_pct']:.2f}%) | ATE = {res['ate_rmse']:>6.2f}m", flush=True)

    n = len(DIVERSE_TEST_SPECS)
    print(f"AVERAGE DRIFT: {total_drift/n:.2f} m | AVERAGE ATE: {total_ate/n:.2f} m", flush=True)

if __name__ == "__main__":
    main()
