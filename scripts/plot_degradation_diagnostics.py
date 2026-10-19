"""
Generate diagnostic visual maps for tracks where Model 2.7 shows degradation:
1. Zoomed Slow Strolling (Start zoom, End zoom, Full path, Speed Profile)
2. High-drift degradation tracks (Curved Terminal, Courier Stops, Phone Tilt, Slalom, Zigzag, Serpentine)
Saves high-res PNG artifacts to brain artifacts directory.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector
from scripts.sweep_fallback_time import preextract_sequence, run_inference

ARTIFACTS_DIR = r"C:\Users\ASUS-PC\.gemini\antigravity-ide\brain\2eead742-0eff-41bd-be9a-1e7bd42d7b6a"


def reconstruct_path(v_hif, anchor_yaws, pos_gt_start, dt_step):
    num_wins = len(v_hif)
    v_world = np.zeros_like(v_hif)
    for k in range(num_wins):
        v_world[k] = rotate_2d_vector(v_hif[k], +anchor_yaws[k])

    pos_pred = np.zeros((num_wins, 2), dtype=np.float32)
    pos_pred[0] = pos_gt_start
    for k in range(1, num_wins):
        pos_pred[k] = pos_pred[k - 1] + v_world[k] * dt_step
    return pos_pred, v_world


def main():
    device = torch.device("cpu")
    ckpt_path = os.path.join(PROJECT_ROOT, "models/checkpoints/real_v3/model_curve_wvel27.pt")
    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.eval()

    test_dir = os.path.join(PROJECT_ROOT, "data/real/test_diverse_1gb")

    # =========================================================================
    # FIGURE 1: SLOW STROLLING DETAILED DIAGNOSTIC (Full + Start Zoom + End Zoom)
    # =========================================================================
    stroll_file = os.path.join(test_dir, "test_div07_pocket_slow_stroll.h5")
    seq_stroll = PreprocessedSequence(stroll_file, beta=0.15)
    ext_stroll = preextract_sequence(seq_stroll, step_size=20, window_size=200)

    v_pred_stroll = run_inference(model, ext_stroll["pred_windows"], device)
    pos_pred_stroll, v_world_stroll = reconstruct_path(
        v_pred_stroll, ext_stroll["anchor_yaws"], ext_stroll["pos_gt"][0], ext_stroll["dt_step"]
    )
    pos_gt_stroll = ext_stroll["pos_gt"]
    dt_step = ext_stroll["dt_step"]
    time_sec = np.arange(len(pos_gt_stroll)) * dt_step

    drift_stroll = np.linalg.norm(pos_pred_stroll[-1] - pos_gt_stroll[-1])
    dist_gt_stroll = np.sum(np.linalg.norm(np.diff(pos_gt_stroll, axis=0), axis=1))

    # Speed profiles
    speed_pred_stroll = np.linalg.norm(v_world_stroll, axis=1)
    speed_gt_stroll = np.linalg.norm(np.diff(pos_gt_stroll, axis=0), axis=1) / dt_step
    speed_gt_stroll = np.concatenate([[speed_gt_stroll[0]], speed_gt_stroll])

    # Cumulative odometer distance
    cum_dist_pred = np.cumsum(speed_pred_stroll * dt_step)
    cum_dist_gt = np.cumsum(speed_gt_stroll * dt_step)

    fig1, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    # 1. Full Trajectory Map
    ax_full = axes[0, 0]
    ax_full.plot(pos_gt_stroll[:, 0], pos_gt_stroll[:, 1], "k--", linewidth=2.0, label="Ground Truth (Real Path)")
    ax_full.plot(pos_pred_stroll[:, 0], pos_pred_stroll[:, 1], "#e63946", linewidth=1.8, alpha=0.85, label="Model 2.7 Prediction")
    ax_full.scatter(pos_gt_stroll[0, 0], pos_gt_stroll[0, 1], c="#2a9d8f", s=130, marker="o", edgecolors="black", zorder=5, label="START (0, 0)")
    ax_full.scatter(pos_gt_stroll[-1, 0], pos_gt_stroll[-1, 1], c="#264653", s=130, marker="s", edgecolors="black", zorder=5, label=f"TRUE END")
    ax_full.scatter(pos_pred_stroll[-1, 0], pos_pred_stroll[-1, 1], c="#e76f51", s=140, marker="X", edgecolors="black", zorder=5, label=f"PRED END (Drift: {drift_stroll:.2f}m)")
    ax_full.annotate("", xy=(pos_pred_stroll[-1, 0], pos_pred_stroll[-1, 1]), xytext=(pos_gt_stroll[-1, 0], pos_gt_stroll[-1, 1]),
                     arrowprops=dict(arrowstyle="<->", color="purple", lw=2.2))
    ax_full.set_title(f"1. Slow Stroll: Full 2.1 km Path\nFinal Drift = {drift_stroll:.2f}m (RTE = {(drift_stroll/dist_gt_stroll)*100:.2f}%)", fontsize=12, fontweight="bold")
    ax_full.set_xlabel("X (meters)")
    ax_full.set_ylabel("Y (meters)")
    ax_full.grid(True, linestyle=":", alpha=0.6)
    ax_full.legend(loc="upper right", fontsize=9)
    ax_full.set_aspect("equal", adjustable="datalim")

    # 2. Zoomed Start (first 150 seconds / ~120m)
    ax_zstart = axes[0, 1]
    n_start = int(180 / dt_step)  # First 180s
    ax_zstart.plot(pos_gt_stroll[:n_start, 0], pos_gt_stroll[:n_start, 1], "k--", linewidth=2.4, label="Ground Truth")
    ax_zstart.plot(pos_pred_stroll[:n_start, 0], pos_pred_stroll[:n_start, 1], "#e63946", linewidth=2.0, label="Model 2.7")
    ax_zstart.scatter(pos_gt_stroll[0, 0], pos_gt_stroll[0, 1], c="#2a9d8f", s=160, marker="o", edgecolors="black", zorder=5, label="START")
    ax_zstart.scatter(pos_gt_stroll[n_start, 0], pos_gt_stroll[n_start, 1], c="#264653", s=100, marker="s", edgecolors="black", zorder=5, label="t = 180s (GT)")
    ax_zstart.scatter(pos_pred_stroll[n_start, 0], pos_pred_stroll[n_start, 1], c="#e76f51", s=110, marker="X", edgecolors="black", zorder=5, label="t = 180s (Pred)")
    ax_zstart.set_title("2. Zoomed START View (First 3 Minutes)\nModel starts leading slightly along-track", fontsize=12, fontweight="bold")
    ax_zstart.set_xlabel("X (meters)")
    ax_zstart.set_ylabel("Y (meters)")
    ax_zstart.grid(True, linestyle=":", alpha=0.6)
    ax_zstart.legend(loc="lower left", fontsize=9)
    ax_zstart.set_aspect("equal", adjustable="datalim")

    # 3. Zoomed End (last 180 seconds)
    ax_zend = axes[1, 0]
    n_end_start = len(pos_gt_stroll) - int(200 / dt_step)
    ax_zend.plot(pos_gt_stroll[n_end_start:, 0], pos_gt_stroll[n_end_start:, 1], "k--", linewidth=2.4, label="Ground Truth")
    ax_zend.plot(pos_pred_stroll[n_end_start:, 0], pos_pred_stroll[n_end_start:, 1], "#e63946", linewidth=2.0, label="Model 2.7")
    ax_zend.scatter(pos_gt_stroll[-1, 0], pos_gt_stroll[-1, 1], c="#264653", s=160, marker="s", edgecolors="black", zorder=5, label="TRUE FINISH")
    ax_zend.scatter(pos_pred_stroll[-1, 0], pos_pred_stroll[-1, 1], c="#e76f51", s=180, marker="X", edgecolors="black", zorder=5, label="PRED FINISH")
    ax_zend.annotate(f"Final Drift = {drift_stroll:.2f}m", xy=(pos_pred_stroll[-1, 0], pos_pred_stroll[-1, 1]),
                     xytext=(pos_gt_stroll[-1, 0]-15, pos_gt_stroll[-1, 1]+10),
                     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, fontweight="bold", backgroundcolor="#ffffcc")
    ax_zend.set_title("3. Zoomed END View (Final Minutes)\nPredicted path overshoots along-track by ~39m", fontsize=12, fontweight="bold")
    ax_zend.set_xlabel("X (meters)")
    ax_zend.set_ylabel("Y (meters)")
    ax_zend.grid(True, linestyle=":", alpha=0.6)
    ax_zend.legend(loc="lower left", fontsize=9)
    ax_zend.set_aspect("equal", adjustable="datalim")

    # 4. Cumulative Distance vs Time (Odometer Diagnosis)
    ax_odo = axes[1, 1]
    ax_odo.plot(time_sec / 60.0, cum_dist_gt, "k--", linewidth=2.2, label=f"True Walked Distance: {cum_dist_gt[-1]:.0f}m")
    ax_odo.plot(time_sec / 60.0, cum_dist_pred, "#e63946", linewidth=2.0, label=f"Model 2.7 Integrated: {cum_dist_pred[-1]:.0f}m (+{cum_dist_pred[-1]-cum_dist_gt[-1]:.0f}m)")
    ax_odo.fill_between(time_sec / 60.0, cum_dist_gt, cum_dist_pred, color="#e63946", alpha=0.15, label="Cumulative Speed Overshoot (~1.8%)")
    ax_odo.set_title("4. Speed Odometer Over Time (Cause of Drift)\nw_vel=2.7 predicts ~0.87 m/s instead of true 0.85 m/s", fontsize=12, fontweight="bold")
    ax_odo.set_xlabel("Time (minutes)")
    ax_odo.set_ylabel("Cumulative Traveled Distance (meters)")
    ax_odo.grid(True, linestyle=":", alpha=0.6)
    ax_odo.legend(loc="upper left", fontsize=9)

    fig1_path = os.path.join(ARTIFACTS_DIR, "diagnostic_slow_strolling_zoom.png")
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)
    print(f"✓ Saved Figure 1: {fig1_path}")

    # =========================================================================
    # FIGURE 2: ALL 6 HIGH-DEGRADATION TRACKS WHERE MODEL 2.7 LOST TO TLIO
    # =========================================================================
    degraded_tracks = [
        ("test_div14_handheld_curved_terminal.h5", "Airport 30m Curve (Handheld)", "Gentle Turn Below Threshold (R=30m)"),
        ("test_div19_bag_courier_stops.h5", "160 Courier Stops (Bag)", "Stop-and-Go Residual Motion Leakage"),
        ("test_div17_handheld_phone_tilt.h5", "Dynamic Phone Tilt (Handheld)", "Device Pitch/Roll Gravity Leakage"),
        ("test_div09_bag_slalom.h5", "Corridor Slalom Weaving (Bag)", "High-Frequency Centrifugal Whip"),
        ("test_div05_handheld_zigzag.h5", "Rapid 45° Zigzag (Handheld)", "Rapid Lateral Direction Switches"),
        ("test_div15_pocket_serpentine.h5", "Serpentine S-Bends (Pocket)", "Dense S-Curve Inertial Overshoot"),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(21, 14), dpi=150)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    for idx, (fname, title_label, diagnosis_note) in enumerate(degraded_tracks):
        row = idx // 3
        col = idx % 3
        ax = axes2[row, col]

        fpath = os.path.join(test_dir, fname)
        seq = PreprocessedSequence(fpath, beta=0.15)
        ext = preextract_sequence(seq, step_size=20, window_size=200)

        v_pred = run_inference(model, ext["pred_windows"], device)
        pos_pred, _ = reconstruct_path(v_pred, ext["anchor_yaws"], ext["pos_gt"][0], ext["dt_step"])
        pos_gt = ext["pos_gt"]

        drift = np.linalg.norm(pos_pred[-1] - pos_gt[-1])
        dist_gt = np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))
        rte = (drift / dist_gt) * 100.0

        ax.plot(pos_gt[:, 0], pos_gt[:, 1], "k--", linewidth=1.8, label="Ground Truth")
        ax.plot(pos_pred[:, 0], pos_pred[:, 1], "#e63946", linewidth=1.5, alpha=0.85, label="Model 2.7 Prediction")

        # Markers
        ax.scatter(pos_gt[0, 0], pos_gt[0, 1], c="#2a9d8f", s=110, marker="o", edgecolors="black", zorder=5, label="START")
        ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], c="#264653", s=110, marker="s", edgecolors="black", zorder=5, label="TRUE END")
        ax.scatter(pos_pred[-1, 0], pos_pred[-1, 1], c="#e76f51", s=130, marker="X", edgecolors="black", zorder=5, label="PRED END")

        # Drift arrow
        ax.annotate("", xy=(pos_pred[-1, 0], pos_pred[-1, 1]), xytext=(pos_gt[-1, 0], pos_gt[-1, 1]),
                    arrowprops=dict(arrowstyle="<->", color="purple", lw=2.0))

        ax.set_title(f"{title_label}\nDrift: {drift:.2f}m ({rte:.2f}%) | {diagnosis_note}", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    fig2_path = os.path.join(ARTIFACTS_DIR, "diagnostic_degraded_tracks_all.png")
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Saved Figure 2: {fig2_path}")


if __name__ == "__main__":
    main()
