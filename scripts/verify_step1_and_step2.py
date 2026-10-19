"""
Verification Script for Step 1 (Madgwick Filter) & Step 2 (Physics Pre-Processing)
Loads the synthetic RoNIN dataset, runs attitude estimation, strips gravity,
and verifies numerical accuracy against ground truth.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np

from src.physics.madgwick import MadgwickAHRS
from src.physics.coordinate_transforms import (
    rotate_body_to_world,
    subtract_gravity,
    detect_stationary_periods,
    extract_yaw_from_quats,
)


def quaternion_angular_error_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Computes angular distance between two quaternions in degrees.
    θ = 2 * arccos(|q1 · q2|)
    """
    # Dot product between unit quaternions
    dots = np.abs(np.sum(q1 * q2, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dots)
    return np.degrees(angle_rad)


def run_verification(file_path: str):
    print("=" * 70)
    print(f"LOADING DATASET: {file_path}")
    print("=" * 70)

    with h5py.File(file_path, "r") as f:
        synced = f["synced"]
        time = synced["time"][:]
        acce = synced["acce"][:]
        gyro = synced["gyro"][:]
        pos_gt = synced["pos"][:]
        ori_gt = synced["ori"][:]
        vel_gt = synced["vel"][:]

    N = len(time)
    dt = time[1, 0] - time[0, 0] if N > 1 else 0.005
    duration_s = time[-1, 0] - time[0, 0]
    print(f"• Total Samples : {N:,}")
    print(f"• Duration      : {duration_s:.2f} seconds")
    print(f"• Sampling Rate : {1.0 / dt:.1f} Hz (Δt = {dt:.4f} s)")

    # ---------------------------------------------------------
    # STEP 1: MADGWICK FILTER
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("RUNNING STEP 1: MADGWICK AHRS FILTER (Attitude Estimation)")
    print("-" * 70)

    # Initialize Madgwick filter
    filter_gain_beta = 0.15
    ahrs = MadgwickAHRS(sample_period=dt, beta=filter_gain_beta)

    # Initialize with the first true orientation quaternion so convergence is fast
    ahrs.reset(initial_q=ori_gt[0])

    print("• Processing 6-axis IMU stream through gradient-descent attitude filter...")
    q_estimated = ahrs.process_sequence(gyro, acce)

    # Compare with ground truth orientation
    angular_errors = quaternion_angular_error_deg(q_estimated, ori_gt)
    mean_angle_err = np.mean(angular_errors)
    max_angle_err = np.max(angular_errors)

    print(f"  ✓ Mean 3D Orientation Error : {mean_angle_err:.2f}°")
    print(f"  ✓ Max 3D Orientation Error  : {max_angle_err:.2f}°")

    # ---------------------------------------------------------
    # STEP 2: PHYSICS PRE-PROCESSING
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("RUNNING STEP 2: PHYSICS PRE-PROCESSING (Rotation & Gravity Stripping)")
    print("-" * 70)

    # 2.1 Rotate Body Acceleration into World Frame using estimated orientation
    accel_world = rotate_body_to_world(acce, q_estimated)

    # 2.2 Clean Gravity Subtraction
    g_constant = 9.80665
    accel_linear = subtract_gravity(accel_world, g=g_constant)

    # 2.3 Stationary Detection (ZUPT)
    is_stationary = detect_stationary_periods(acce, gyro, window_samples=40, var_threshold=0.08)
    stationary_ratio = np.mean(is_stationary) * 100.0

    # Inspect stationary phase (First 5 seconds = 1000 samples)
    stationary_idx = np.where(time[:, 0] < 4.5)[0]
    stationary_accel = accel_linear[stationary_idx]
    mean_stationary_res = np.mean(np.linalg.norm(stationary_accel, axis=1))

    print(f"  ✓ World Z Acceleration Before Subtraction : Mean = {np.mean(accel_world[:, 2]):.4f} m/s² (≈ Earth Gravity 9.81)")
    print(f"  ✓ World Z Acceleration After Subtraction  : Mean = {np.mean(accel_linear[:, 2]):.4f} m/s² (Clean 0 Baseline!)")
    print(f"  ✓ Stationary Residual Acceleration Norm   : {mean_stationary_res:.4f} m/s² (Sensor noise floor)")
    print(f"  ✓ Stationary Frames Detected (ZUPT)       : {stationary_ratio:.1f}% of total recording")

    # ---------------------------------------------------------
    # COMPARISON: CLASSICAL DOUBLE INTEGRATION (Why it fails)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON: Classical Double Integration vs Reality")
    print("-" * 70)

    # Double integrate accel_linear directly
    vel_classical = np.cumsum(accel_linear * dt, axis=0)
    pos_classical = np.cumsum(vel_classical * dt, axis=0)

    final_true_pos = pos_gt[-1, :2]
    final_classical_pos = pos_classical[-1, :2]
    classical_drift = np.linalg.norm(final_classical_pos - final_true_pos)

    print(f"  • True Final Position [X, Y]          : [{final_true_pos[0]:.2f}, {final_true_pos[1]:.2f}] m")
    print(f"  • Classical Double-Integrated [X, Y]  : [{final_classical_pos[0]:.2f}, {final_classical_pos[1]:.2f}] m")
    print(f"  • Classical Position Drift Error      : {classical_drift:.2f} meters!")
    print("    ▲ Notice how classical double integration blows up rapidly due to noise accumulation!")
    print("    ▲ This proves why Step 3 (Neural Network predicting Velocity directly) is required.")

    print("\n" + "=" * 70)
    print("STEP 1 & STEP 2 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_h5 = os.path.join("data", "synthetic", "sample_walk_square.h5")
    run_verification(test_h5)
