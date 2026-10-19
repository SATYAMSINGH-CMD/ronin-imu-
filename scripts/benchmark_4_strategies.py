"""
Benchmark the 4 Strategies:
1. 2.7 All Over
2. 2.1 All Over
3. 2.1 on Real World (Straight) + 2.7 on Curve (Figure-8)
4. 2.7 on Real World (Straight) + 2.1 on Curve (Figure-8)
All models have w_dir = 2.5.
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame

def evaluate_strategy(
    strategy_mode: str,
    model_21,
    model_27,
    seq,
    device,
    turn_thresh: float = 0.25,
    step_size: int = 10,
    window_size: int = 200,
):
    total_samples = seq.N
    dt = seq.dt
    dt_step = float(step_size * dt)

    pred_windows = []
    anchor_yaws = []
    window_indices = []
    turn_rates = []

    for start_idx in range(0, total_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        accel_win = seq.accel_linear_world[start_idx:end_idx]
        gyro_win = seq.gyro_world[start_idx:end_idx]
        anchor_yaw = float(seq.yaw_angles[start_idx])

        a_hif, w_hif = project_to_heading_independent_frame(accel_win, gyro_win, anchor_yaw)
        features = np.concatenate([a_hif, w_hif], axis=1).T
        pred_windows.append(features)
        anchor_yaws.append(anchor_yaw)
        window_indices.append(start_idx)
        turn_rates.append(np.mean(np.abs(w_hif[:, 2])))

    pred_np = np.array(pred_windows, dtype=np.float32)
    turn_rates_np = np.array(turn_rates, dtype=np.float32)
    v_hif_preds = np.zeros((len(pred_np), 2), dtype=np.float32)

    with torch.no_grad():
        for b_start in range(0, len(pred_np), 512):
            b_end = min(b_start + 512, len(pred_np))
            b_tensor = torch.from_numpy(pred_np[b_start:b_end]).to(device)
            b_turns = turn_rates_np[b_start:b_end]

            if strategy_mode == "2.7_all_over":
                v_out, _ = model_27(b_tensor)
                v_hif_preds[b_start:b_end] = v_out.cpu().numpy()

            elif strategy_mode == "2.1_all_over":
                v_out, _ = model_21(b_tensor)
                v_hif_preds[b_start:b_end] = v_out.cpu().numpy()

            elif strategy_mode == "2.1_straight_2.7_curve":
                v_27, _ = model_27(b_tensor)
                v_21, _ = model_21(b_tensor)
                # alpha = 1 for curves (uses 2.7), alpha = 0 for straight (uses 2.1)
                alpha = np.clip((b_turns - 0.15) / (turn_thresh - 0.15), 0.0, 1.0)[:, None]
                v_blended = alpha * v_27.cpu().numpy() + (1.0 - alpha) * v_21.cpu().numpy()
                v_hif_preds[b_start:b_end] = v_blended

            elif strategy_mode == "2.7_straight_2.1_curve":
                v_27, _ = model_27(b_tensor)
                v_21, _ = model_21(b_tensor)
                # alpha = 1 for curves (uses 2.1), alpha = 0 for straight (uses 2.7)
                alpha = np.clip((b_turns - 0.15) / (turn_thresh - 0.15), 0.0, 1.0)[:, None]
                v_blended = alpha * v_21.cpu().numpy() + (1.0 - alpha) * v_27.cpu().numpy()
                v_hif_preds[b_start:b_end] = v_blended

    v_world_preds = np.zeros_like(v_hif_preds)
    for i in range(len(v_hif_preds)):
        v_world_preds[i] = rotate_2d_vector(v_hif_preds[i], +anchor_yaws[i])

    pos_pred = np.zeros((len(v_world_preds), 2), dtype=np.float32)
    pos_pred[0] = seq.pos_gt[window_indices[0], :2]
    for i in range(1, len(v_world_preds)):
        pos_pred[i] = pos_pred[i - 1] + v_world_preds[i] * dt_step

    pos_gt = np.array([seq.pos_gt[idx, :2] for idx in window_indices])
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)

    traj_dist = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    final_drift = float(errors[-1])
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    rte_pct = (final_drift / traj_dist) * 100.0 if traj_dist > 0 else 0.0

    return {
        "final_drift": final_drift,
        "ate_rmse": ate_rmse,
        "rte_pct": rte_pct,
        "traj_dist": traj_dist,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_21 = os.path.join(PROJECT_ROOT, "models/checkpoints/ronin_impl1_wvel21.pt")
    ckpt_27 = os.path.join(PROJECT_ROOT, "models/checkpoints/ronin_impl1_wvel27.pt")

    model_21 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    model_21.load_state_dict(torch.load(ckpt_21, map_location=device)["model_state_dict"])
    model_21.eval()

    model_27 = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    model_27.load_state_dict(torch.load(ckpt_27, map_location=device)["model_state_dict"])
    model_27.eval()

    test_seqs = [
        ("Pocket Loop (test_sub01)", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5"),
        ("Handheld Mixed (test_sub02)", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5"),
        ("Bag Corridor (test_sub03)", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5"),
        ("Curved Figure-8 (test_sub04)", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5"),
    ]

    strategies = [
        ("Strategy 1: 2.7 All Over", "2.7_all_over"),
        ("Strategy 2: 2.1 All Over", "2.1_all_over"),
        ("Strategy 3: 2.1 Straight + 2.7 Curve", "2.1_straight_2.7_curve"),
        ("Strategy 4: 2.7 Straight + 2.1 Curve", "2.7_straight_2.1_curve"),
    ]

    # Pre-load sequences
    loaded_seqs = [(name, PreprocessedSequence(os.path.join(PROJECT_ROOT, path), beta=0.15)) for name, path in test_seqs]

    results = {strat_name: {} for strat_name, _ in strategies}

    print("=" * 115)
    print("BENCHMARKING 4 STRATEGIES ACROSS ALL TEST ENVIRONMENTS")
    print("=" * 115)

    for strat_label, strat_key in strategies:
        print(f"\nEvaluating {strat_label}...")
        total_drift = 0.0
        total_ate = 0.0
        for seq_name, seq in loaded_seqs:
            metrics = evaluate_strategy(strat_key, model_21, model_27, seq, device)
            results[strat_label][seq_name] = metrics
            total_drift += metrics["final_drift"]
            total_ate += metrics["ate_rmse"]
            print(f"  • {seq_name:<28}: Final Drift = {metrics['final_drift']:>6.2f}m ({metrics['rte_pct']:>4.2f}%) | ATE = {metrics['ate_rmse']:>6.2f}m")
        results[strat_label]["avg_drift"] = total_drift / len(loaded_seqs)
        results[strat_label]["avg_ate"] = total_ate / len(loaded_seqs)

    # Print Comprehensive Comparison Tables
    print("\n" + "=" * 115)
    print("FINAL DRIFT (ENDPOINT ERROR) COMPARISON TABLE")
    print("=" * 115)
    header = f"{'Strategy Name':<38} | {'Pocket Loop':<14} | {'Handheld':<14} | {'Bag Corridor':<14} | {'Figure-8':<14} | {'Overall Average':<15}"
    print(header)
    print("-" * 115)
    for strat_label, _ in strategies:
        p_drift = results[strat_label]["Pocket Loop (test_sub01)"]["final_drift"]
        h_drift = results[strat_label]["Handheld Mixed (test_sub02)"]["final_drift"]
        b_drift = results[strat_label]["Bag Corridor (test_sub03)"]["final_drift"]
        f_drift = results[strat_label]["Curved Figure-8 (test_sub04)"]["final_drift"]
        avg_drift = results[strat_label]["avg_drift"]
        print(f"{strat_label:<38} | {p_drift:>6.2f}m ({results[strat_label]['Pocket Loop (test_sub01)']['rte_pct']:.2f}%) | {h_drift:>6.2f}m ({results[strat_label]['Handheld Mixed (test_sub02)']['rte_pct']:.2f}%) | {b_drift:>6.2f}m ({results[strat_label]['Bag Corridor (test_sub03)']['rte_pct']:.2f}%) | {f_drift:>6.2f}m ({results[strat_label]['Curved Figure-8 (test_sub04)']['rte_pct']:.2f}%) | {avg_drift:>7.2f}m")

    print("\n" + "=" * 115)
    print("ATE RMSE (CONTINUOUS JOURNEY ERROR) COMPARISON TABLE")
    print("=" * 115)
    header_ate = f"{'Strategy Name':<38} | {'Pocket Loop':<14} | {'Handheld':<14} | {'Bag Corridor':<14} | {'Figure-8':<14} | {'Overall Average':<15}"
    print(header_ate)
    print("-" * 115)
    for strat_label, _ in strategies:
        p_ate = results[strat_label]["Pocket Loop (test_sub01)"]["ate_rmse"]
        h_ate = results[strat_label]["Handheld Mixed (test_sub02)"]["ate_rmse"]
        b_ate = results[strat_label]["Bag Corridor (test_sub03)"]["ate_rmse"]
        f_ate = results[strat_label]["Curved Figure-8 (test_sub04)"]["ate_rmse"]
        avg_ate = results[strat_label]["avg_ate"]
        print(f"{strat_label:<38} | {p_ate:>7.2f}m       | {h_ate:>7.2f}m       | {b_ate:>7.2f}m       | {f_ate:>7.2f}m       | {avg_ate:>7.2f}m")

    print("=" * 115)

if __name__ == "__main__":
    main()
