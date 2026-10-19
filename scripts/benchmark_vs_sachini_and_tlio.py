"""
Benchmark Our Models (0.23 & 0.35) vs Sachini (Official RoNIN Baseline) & CathIAS (TLIO)
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
from scripts.sweep_w_vel import evaluate_sequence

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "Sachini (RoNIN Baseline)": "models/checkpoints/ronin_resnet_best.pt",
        "Our Model (Thresh = 0.23)": "models/checkpoints/ronin_dual_regime_thresh_23.pt",
        "Our Model (Thresh = 0.35)": "models/checkpoints/ronin_dual_regime_thresh_35.pt",
    }
    
    loaded_models = {}
    for name, path in models.items():
        full_path = os.path.join(PROJECT_ROOT, path)
        m = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        ckpt = torch.load(full_path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        loaded_models[name] = m

    # 1. Evaluate on 4 Original Holdout Sequences
    print("=" * 100)
    print("BENCHMARK 1: ORIGINAL 4 HOLDOUT TEST SEQUENCES (~200 MB)")
    print("=" * 100)
    
    holdouts = [
        ("Pocket Loop", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "pocket"),
        ("Handheld Walk", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "handheld"),
        ("Bag Corridor", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "bag"),
        ("Figure-8 Track", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "pocket"),
    ]
    
    holdout_results = []
    for desc, path, pose in holdouts:
        fpath = os.path.join(PROJECT_ROOT, path)
        seq = PreprocessedSequence(fpath, beta=0.15)
        res_row = {"name": desc, "pose": pose, "dist": seq.pos_gt[-1] - seq.pos_gt[0]}
        print(f"Testing {desc}...")
        for mname, m in loaded_models.items():
            r = evaluate_sequence(m, seq, device)
            res_row[mname] = r
            print(f"  -> {mname}: Drift = {r['final_drift']:.2f}m ({r['rte_pct']:.2f}%) | ATE = {r['ate_rmse']:.2f}m")
        holdout_results.append(res_row)

    # 2. Evaluate on 10 Diverse Sequences (500 MB)
    print("\n" + "=" * 100)
    print("BENCHMARK 2: 10 DIVERSE REAL-WORLD SEQUENCES (500 MB)")
    print("=" * 100)
    
    div_dir = os.path.join(PROJECT_ROOT, "data/real/test_diverse_500mb")
    from scripts.generate_and_eval_500mb import DIVERSE_TEST_SPECS
    
    div_results = []
    for spec in DIVERSE_TEST_SPECS:
        fpath = os.path.join(div_dir, f"{spec['id']}.h5")
        seq = PreprocessedSequence(fpath, beta=0.15)
        res_row = {"name": spec["desc"], "pose": spec["pose"]}
        print(f"Testing {spec['desc']} [{spec['pose']}]...")
        for mname, m in loaded_models.items():
            r = evaluate_sequence(m, seq, device)
            res_row[mname] = r
            print(f"  -> {mname}: Drift = {r['final_drift']:.2f}m ({r['rte_pct']:.2f}%)")
        div_results.append(res_row)

if __name__ == "__main__":
    run_benchmarks()
