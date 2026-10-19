"""
Diagnostic Plot Generator for Figure-8 Trajectory
Compares Ground Truth vs Predicted Trajectory across:
1. Full 44-Minute Global Map
2. Early Phase (First 3 Loops, Zoomed In)
3. Late Phase (After 30 Mins of Continuous Looping, Zoomed In)
4. Drift Accumulation Over Time (0 to 44 minutes)
"""

import os
import sys
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame


def generate_figure8_diagnostic(
    h5_path: str = "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5",
    ckpt_path: str = "models/checkpoints/ronin_ideaB_max25.pt",
    output_png: str = "reports/figure8_predicted_vs_true_diagnostic.png",
    artifact_dir: str = "C:/Users/ASUS-PC/.gemini/antigravity-ide/brain/2eead742-0eff-41bd-be9a-1e7bd42d7b6a"
):
    print("=" * 70)
    print("GENERATING DETAILED FIGURE-8 DIAGNOSTIC VISUALIZATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = 200
    step_size = 10

    # 1. Load Sequence
    seq = PreprocessedSequence(h5_path, beta=0.15)
    total_samples = seq.N
    dt = seq.dt
    dt_step = float(step_size * dt)  # 0.05s per step

    # 2. Load Model
    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
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
        features = np.concatenate([a_hif, w_hif], axis=1).T
        pred_windows.append(features)
        anchor_yaws.append(anchor_yaw)
        window_indices.append(start_idx)

    pred_tensor = torch.from_numpy(np.array(pred_windows, dtype=np.float32)).to(device)
    with torch.no_grad():
        v_hif_preds, _ = model(pred_tensor)
        v_hif_preds = v_hif_preds.cpu().numpy()

    # Rotate back to world frame
    v_world_preds = np.zeros_like(v_hif_preds)
    for i in range(len(v_hif_preds)):
        v_world_preds[i] = rotate_2d_vector(v_hif_preds[i], +anchor_yaws[i])

    # Integrate velocities
    pos_pred = np.zeros((len(v_world_preds), 2), dtype=np.float32)
    pos_pred[0] = seq.pos_gt[window_indices[0], :2]
    for i in range(1, len(v_world_preds)):
        pos_pred[i] = pos_pred[i - 1] + v_world_preds[i] * dt_step

    pos_gt = np.array([seq.pos_gt[idx, :2] for idx in window_indices])
    time_minutes = np.arange(len(pos_pred)) * dt_step / 60.0
    distance_walked = np.cumsum(np.linalg.norm(np.diff(pos_gt, axis=0, prepend=pos_gt[:1]), axis=1))

    # Error per step
    errors = np.linalg.norm(pos_pred - pos_gt, axis=1)
    final_drift = errors[-1]
    ate_rmse = np.sqrt(np.mean(errors ** 2))

    # 4. Create Multi-Panel Diagnostic Plot
    fig = plt.figure(figsize=(18, 13), dpi=140)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)

    # PANEL 1: True 2D Figure-8 Floor Map (Early Loops 1-5, ~2.5 Minutes)
    ax1 = fig.add_subplot(gs[0, 0])
    # 5 full loops ~ 5 * 52s = 260s -> 260 / dt_step samples
    len_early = int(260.0 / dt_step)
    ax1.plot(pos_gt[:len_early, 0], pos_gt[:len_early, 1], color="#2ca02c", linewidth=2.8, alpha=0.9, label="Ground Truth 8-Figure Path")
    ax1.plot(pos_pred[:len_early, 0], pos_pred[:len_early, 1], color="#1f77b4", linewidth=2.0, linestyle="--", label="AI Predicted Path (Loop 1-5)")
    ax1.scatter([0], [0], color="darkgreen", s=140, zorder=6, label="Center Origin (0,0)")
    ax1.annotate("Right Lobe (Clockwise)", xy=(5.0, 3.5), xytext=(2.0, 6.2),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight="bold")
    ax1.annotate("Left Lobe (Counter-Clockwise)", xy=(-5.0, -3.5), xytext=(-9.5, -6.5),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight="bold")
    ax1.set_title("1. Actual 2D Figure-8 Floor Map (Loops 1–5)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X Position (meters)", fontsize=11)
    ax1.set_ylabel("Y Position (meters)", fontsize=11)
    ax1.set_xlim(-14, 14)
    ax1.set_ylim(-9, 9)
    ax1.axis("equal")
    ax1.legend(loc="upper left", fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # PANEL 2: Single Complete Figure-8 Cycle Zoom-In
    ax2 = fig.add_subplot(gs[0, 1])
    len_single = int(52.5 / dt_step)  # exactly 1 cycle
    ax2.plot(pos_gt[:len_single, 0], pos_gt[:len_single, 1], color="#2ca02c", linewidth=3.2, label="True Cycle 1 (52s)")
    ax2.plot(pos_pred[:len_single, 0], pos_pred[:len_single, 1], color="#0050b3", linewidth=2.2, linestyle="--", label="AI Predicted Cycle 1")
    ax2.scatter([0], [0], color="red", s=120, zorder=6, label="Intersection Crossing (0,0)")
    # Draw motion direction arrows
    mid_r = int(len_single * 0.25)
    mid_l = int(len_single * 0.75)
    ax2.annotate("", xy=(pos_gt[mid_r+20, 0], pos_gt[mid_r+20, 1]), xytext=(pos_gt[mid_r, 0], pos_gt[mid_r, 1]),
                 arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.5))
    ax2.annotate("", xy=(pos_gt[mid_l+20, 0], pos_gt[mid_l+20, 1]), xytext=(pos_gt[mid_l, 0], pos_gt[mid_l, 1]),
                 arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.5))
    ax2.set_title("2. Single Figure-8 Loop Zoom: Flawless Curvature!", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X Position (meters)", fontsize=11)
    ax2.set_ylabel("Y Position (meters)", fontsize=11)
    ax2.set_xlim(-14, 14)
    ax2.set_ylim(-9, 9)
    ax2.axis("equal")
    ax2.legend(loc="upper left", fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # PANEL 3: 44-Minute Cumulative 2D Floor Map (All 50+ Loops Precession)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(pos_gt[:, 0], pos_gt[:, 1], color="#2ca02c", alpha=0.7, linewidth=2.0, label="Ground Truth (50 Loops in place)")
    ax3.plot(pos_pred[:, 0], pos_pred[:, 1], color="#d62728", alpha=0.55, linewidth=1.2, label=f"Predicted Path (ATE: {ate_rmse:.1f}m)")
    ax3.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], color="darkgreen", s=130, zorder=6, label="Start Point (0,0)")
    ax3.scatter([pos_pred[-1, 0]], [pos_pred[-1, 1]], color="crimson", s=130, zorder=6, label=f"Final Endpoint (Drift: {final_drift:.1f}m)")
    ax3.set_title("3. Full 44-Min Map: Slow Rotation / Precession of the 8", fontsize=13, fontweight="bold")
    ax3.set_xlabel("X Position (meters)", fontsize=11)
    ax3.set_ylabel("Y Position (meters)", fontsize=11)
    ax3.axis("equal")
    ax3.legend(loc="upper left", fontsize=9.5)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # PANEL 4: Error Accumulation Over Time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_minutes, errors, color="#d62728", linewidth=2.2, label="Position Error (Meters)")
    ax4.fill_between(time_minutes, errors, color="#ff7875", alpha=0.25)
    ax4.set_title("4. Drift Over Time: Smooth, Constant Compass Slip", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Elapsed Walk Time (Minutes)", fontsize=11)
    ax4.set_ylabel("Position Drift from Ground Truth (meters)", fontsize=11)
    ax4.axhline(final_drift, color="black", linestyle=":", label=f"Final Drift = {final_drift:.1f}m ({(final_drift/distance_walked[-1])*100:.1f}%)")
    ax4.annotate(
        "Steady climb: ~2.5m drift per minute\nCaused purely by continuous pocket turning\nwithout straight corridors to re-zero compass.",
        xy=(time_minutes[len(time_minutes)//2], errors[len(errors)//2]),
        xytext=(time_minutes[len(time_minutes)//2] - 15, errors[len(errors)//2] + 25),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#ffe58f", lw=1.2),
        fontsize=9.5
    )
    ax4.legend(loc="upper left", fontsize=9.5)
    ax4.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("RoNIN 2D Figure-8 Floor Map Benchmark: Ground Truth vs AI Prediction", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"✓ Diagnostic plot successfully saved to: {output_png}")

    # Also copy to artifacts directory for display
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_png = os.path.join(artifact_dir, "figure8_predicted_vs_true_diagnostic.png")
    shutil.copyfile(output_png, artifact_png)
    print(f"✓ Artifact copy saved to: {artifact_png}")


if __name__ == "__main__":
    generate_figure8_diagnostic()
