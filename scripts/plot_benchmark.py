"""
Generates visual benchmark plots for Step 1 (Madgwick) and Step 2 (Gravity Subtraction).
Saves figures to reports/step1_step2_verification.png
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.physics.madgwick import MadgwickAHRS
from src.physics.coordinate_transforms import (
    rotate_body_to_world,
    subtract_gravity,
    detect_stationary_periods,
)
from scripts.verify_step1_and_step2 import quaternion_angular_error_deg


def generate_benchmark_plots(h5_path: str, output_image_path: str):
    with h5py.File(h5_path, "r") as f:
        synced = f["synced"]
        time = synced["time"][:, 0]
        acce = synced["acce"][:]
        gyro = synced["gyro"][:]
        pos_gt = synced["pos"][:]
        ori_gt = synced["ori"][:]

    dt = time[1] - time[0]

    # Step 1: Madgwick
    ahrs = MadgwickAHRS(sample_period=dt, beta=0.15)
    ahrs.reset(initial_q=ori_gt[0])
    q_est = ahrs.process_sequence(gyro, acce)
    ang_errors = quaternion_angular_error_deg(q_est, ori_gt)

    # Step 2: Gravity Subtraction
    accel_world = rotate_body_to_world(acce, q_est)
    accel_linear = subtract_gravity(accel_world, g=9.80665)

    # Classical double integration
    vel_classical = np.cumsum(accel_linear * dt, axis=0)
    pos_classical = np.cumsum(vel_classical * dt, axis=0)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    # 1. 2D Trajectory: True vs Classical Double Integration
    ax1 = axes[0]
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1], "g-", linewidth=2.5, label="Ground Truth Path (10m x 10m)")
    ax1.plot(pos_classical[:, 0], pos_classical[:, 1], "r--", linewidth=1.8, label="Classical Double Integration (Drifting)")
    ax1.scatter([pos_gt[0, 0]], [pos_gt[0, 1]], color="blue", s=100, zorder=5, label="Start / Finish (0, 0)")
    ax1.set_title("1. Trajectory Drift (55 seconds)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X Position (meters)", fontsize=11)
    ax1.set_ylabel("Y Position (meters)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axis("equal")

    # 2. Gravity Subtraction
    ax2 = axes[1]
    ax2.plot(time, accel_world[:, 2], color="orange", alpha=0.7, label="World Z (With 9.81 m/s² Gravity)")
    ax2.plot(time, accel_linear[:, 2], color="blue", alpha=0.8, label="Linear Z (Clean Gravity Stripped)")
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("2. Step 2: Clean Gravity Subtraction", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Acceleration (m/s²)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Attitude Estimation Error
    ax3 = axes[2]
    ax3.plot(time, ang_errors, color="purple", linewidth=1.5, label="3D Orientation Error")
    ax3.axhline(np.mean(ang_errors), color="red", linestyle="--", label=f"Mean Error: {np.mean(ang_errors):.2f}°")
    ax3.set_title("3. Step 1: Madgwick Attitude Accuracy", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.set_ylabel("Error (degrees)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    plt.close()
    print(f"Benchmark plot successfully saved to: {output_image_path}")


if __name__ == "__main__":
    test_h5 = os.path.join("data", "synthetic", "sample_walk_square.h5")
    out_img = os.path.join("reports", "step1_step2_verification.png")
    generate_benchmark_plots(test_h5, out_img)
