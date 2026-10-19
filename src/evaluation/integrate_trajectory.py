"""
Trajectory Integration and Evaluation Engine for RoNIN
Takes trained 1D-ResNet velocity predictions, rotates them back from
Heading-Independent Frame (HIF) to World Frame, integrates once (p = p + v * dt),
and evaluates Absolute Trajectory Error (ATE) against ground truth.
Generates visual floor plan map: reports/trajectory_reconstruction.png
"""

import os
import sys
from typing import Tuple, Dict

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame


def compute_ate_and_rte(
    pos_pred: np.ndarray,
    pos_gt: np.ndarray,
    dt: float = 0.05
) -> Dict[str, float]:
    """
    Computes standard trajectory evaluation metrics:
    - ATE: Absolute Trajectory Error (RMSE in meters)
    - Final Distance Error (meters)
    - Total Trajectory Length (meters)
    - RTE: Relative Trajectory Error (% of total distance walked)
    """
    # 2D errors (horizontal floor plane)
    errors = np.linalg.norm(pos_pred[:, :2] - pos_gt[:, :2], axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    final_drift = float(errors[-1])

    # Total distance walked
    step_dists = np.linalg.norm(np.diff(pos_gt[:, :2], axis=0), axis=1)
    total_distance = float(np.sum(step_dists))

    rte_percent = float((final_drift / total_distance) * 100.0) if total_distance > 0 else 0.0

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "final_drift": final_drift,
        "total_distance": total_distance,
        "rte_percent": rte_percent,
    }


def evaluate_trajectory(
    h5_path: str,
    checkpoint_path: str,
    output_plot_path: str = "reports/trajectory_reconstruction.png",
    window_size: int = 200,
    step_size: int = 10,
) -> Dict[str, float]:
    print("=" * 70)
    print("PHASE 5: TRAJECTORY RECONSTRUCTION & FLOOR MAP EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Preprocessed Sequence
    print(f"• Loading Sequence       : {os.path.basename(h5_path)}")
    seq = PreprocessedSequence(h5_path, beta=0.15)
    total_samples = seq.N
    dt = seq.dt

    # 2. Load Model Checkpoint
    print(f"• Loading Trained Model  : {checkpoint_path}")
    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Sliding Window Inference
    pred_windows = []
    anchor_yaws = []
    window_indices = []

    for start_idx in range(0, total_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        accel_win = seq.accel_linear_world[start_idx:end_idx]
        gyro_win = seq.gyro_world[start_idx:end_idx]
        anchor_yaw = float(seq.yaw_angles[start_idx])

        a_hif, w_hif = project_to_heading_independent_frame(accel_win, gyro_win, anchor_yaw)
        features = np.concatenate([a_hif, w_hif], axis=1).T  # (6, window_size)
        pred_windows.append(features)
        anchor_yaws.append(anchor_yaw)
        window_indices.append(start_idx)

    # Batched inference to prevent CPU memory spike
    pred_np = np.array(pred_windows, dtype=np.float32)
    v_hif_list = []
    batch_size = 512

    with torch.no_grad():
        for b_start in range(0, len(pred_np), batch_size):
            b_end = min(b_start + batch_size, len(pred_np))
            b_tensor = torch.from_numpy(pred_np[b_start:b_end]).to(device)
            v_batch, _ = model(b_tensor)
            v_hif_list.append(v_batch.cpu().numpy())

    v_hif_preds = np.concatenate(v_hif_list, axis=0)

    # 4. Rotate Velocities Back from HIF to World Coordinates
    # v_world = rotate_2d(v_hif, +anchor_yaw)
    v_world_preds = np.zeros_like(v_hif_preds)
    for i in range(len(v_hif_preds)):
        v_world_preds[i] = rotate_2d_vector(v_hif_preds[i], +anchor_yaws[i])

    # 5. Integrate Velocities ONCE to reconstruct 2D Position Trajectory
    # dt_step = time duration of each step
    dt_step = float(step_size * dt)  # e.g., 10 * 0.005 = 0.05s
    pos_pred = np.zeros((len(v_world_preds), 2), dtype=np.float32)
    pos_pred[0] = seq.pos_gt[window_indices[0], :2]

    for i in range(1, len(v_world_preds)):
        pos_pred[i] = pos_pred[i - 1] + v_world_preds[i] * dt_step

    # Extract matching Ground Truth positions for each window
    pos_gt_aligned = np.array([seq.pos_gt[idx, :2] for idx in window_indices])

    # 6. Compute Classical Double Integration for Comparison
    vel_classical = np.cumsum(seq.accel_linear_world * dt, axis=0)
    pos_classical = np.cumsum(vel_classical * dt, axis=0)
    pos_classical_aligned = np.array([pos_classical[idx, :2] for idx in window_indices])

    # 7. Compute Metrics
    metrics = compute_ate_and_rte(pos_pred, pos_gt_aligned, dt=dt_step)

    print("\n" + "-" * 70)
    print("TRAJECTORY EVALUATION METRICS")
    print("-" * 70)
    print(f"• Total Distance Walked    : {metrics['total_distance']:.2f} meters")
    print(f"• Absolute Trajectory Error: {metrics['ate_rmse']:.2f} meters (ATE RMSE)")
    print(f"• Mean Position Error      : {metrics['ate_mean']:.2f} meters")
    print(f"• Final Endpoint Drift     : {metrics['final_drift']:.2f} meters")
    print(f"• Relative Trajectory Error: {metrics['rte_percent']:.2f}% (RTE)")

    # 8. Generate Floor Plan Map Plot
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=130)

    # Panel 1: AI Model Path vs Ground Truth (The Accurate Map)
    ax1 = axes[0]
    ax1.plot(pos_gt_aligned[:, 0], pos_gt_aligned[:, 1], "g-", linewidth=3.0, label="Ground Truth (10m x 10m Square)")
    ax1.plot(pos_pred[:, 0], pos_pred[:, 1], "b--", linewidth=2.5, label=f"1D-ResNet AI Path (ATE: {metrics['ate_rmse']:.2f}m)")
    ax1.scatter([pos_gt_aligned[0, 0]], [pos_gt_aligned[0, 1]], color="darkgreen", s=120, zorder=5, label="Start Point (0,0)")
    ax1.scatter([pos_pred[-1, 0]], [pos_pred[-1, 1]], color="red", s=100, zorder=5, label=f"AI Endpoint (Drift: {metrics['final_drift']:.2f}m)")
    ax1.set_title("1. AI Predicted Indoor Trajectory (Single Integration)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X Position (meters)", fontsize=11)
    ax1.set_ylabel("Y Position (meters)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axis("equal")

    # Panel 2: Why AI Wins (AI Path vs Classical Double Integration Disaster)
    ax2 = axes[1]
    ax2.plot(pos_gt_aligned[:, 0], pos_gt_aligned[:, 1], "g-", linewidth=2.5, label="Ground Truth")
    ax2.plot(pos_pred[:, 0], pos_pred[:, 1], "b-", linewidth=2.0, label="1D-ResNet AI (Meter-Accurate)")
    ax2.plot(pos_classical_aligned[:, 0], pos_classical_aligned[:, 1], "r-.", linewidth=1.8, label="Classical Double Integration (Drifting!)")
    ax2.set_title("2. Benchmark: AI vs Classical Double Integration", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X Position (meters)", fontsize=11)
    ax2.set_ylabel("Y Position (meters)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.axis("equal")

    plt.suptitle("RoNIN Production ML Lifecycle — Trajectory Reconstruction Benchmark", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()

    print(f"\n✓ High-resolution floor plan map saved to: {output_plot_path}")
    return metrics


def evaluate_directory(
    dir_path: str,
    checkpoint_path: str,
    output_dir: str = "reports"
) -> Dict[str, Dict[str, float]]:
    """Evaluates all .h5 sequences in a directory against the trained model."""
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if f.endswith((".h5", ".hdf5"))]
    if not files:
        print(f"No .h5 sequences found in {dir_path}")
        return {}

    summary = {}
    print("\n" + "=" * 80)
    print(f"{'Sequence':<36}{'Dist (m)':<12}{'ATE RMSE (m)':<15}{'Final Drift (m)':<18}{'RTE %':<10}")
    print("=" * 80)

    for fpath in files:
        fname = os.path.basename(fpath)
        plot_out = os.path.join(output_dir, f"eval_{os.path.splitext(fname)[0]}.png")
        m = evaluate_trajectory(fpath, checkpoint_path, output_plot_path=plot_out)
        summary[fname] = m
        print(f"{fname:<36}{m['total_distance']:<12.2f}{m['ate_rmse']:<15.2f}{m['final_drift']:<18.2f}{m['rte_percent']:<10.2f}")

    print("=" * 80)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RoNIN Trajectory Reconstructor and Evaluator")
    parser.add_argument("--h5", type=str, default="data/synthetic/sample_walk_square.h5", help="Path to sequence or directory")
    parser.add_argument("--ckpt", type=str, default="models/checkpoints/ronin_resnet_best.pt", help="Model checkpoint path")
    parser.add_argument("--out", type=str, default="reports/trajectory_reconstruction.png", help="Output plot path")
    args = parser.parse_args()

    target = os.path.join(PROJECT_ROOT, args.h5) if not os.path.isabs(args.h5) else args.h5
    ckpt = os.path.join(PROJECT_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt

    if os.path.isdir(target):
        evaluate_directory(target, ckpt, output_dir=os.path.dirname(args.out) or "reports")
    else:
        evaluate_trajectory(h5_path=target, checkpoint_path=ckpt, output_plot_path=args.out)
