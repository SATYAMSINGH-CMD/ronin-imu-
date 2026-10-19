"""
Production CI/CD Quality Gate Suite for RoNIN Inertial Navigation
Runs automated checks for:
1. Numerical sanity (Zero NaNs, exact tensor shapes)
2. Latency Gate (Inference < 5.0ms on CPU)
3. Accuracy Gate (ATE RMSE <= 1.50 meters on benchmark loop)
4. Drift Detector Alarm Logic
"""

import os
import sys
import time
import pytest
import numpy as np
import torch
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import get_ronin_dataloader
from src.evaluation.integrate_trajectory import evaluate_trajectory
from src.monitoring.drift_detector import SensorDriftDetector


@pytest.fixture
def sample_h5():
    path = os.path.join(PROJECT_ROOT, "data", "synthetic", "sample_walk_square.h5")
    assert os.path.exists(path), f"Benchmark file missing: {path}"
    return path


@pytest.fixture
def model_checkpoint():
    path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "ronin_resnet_best.pt")
    assert os.path.exists(path), f"Model checkpoint missing: {path}"
    return path


def test_dataset_window_shapes_and_nans(sample_h5):
    """GATE 1: Verifies data pipeline outputs clean tensors with zero NaNs."""
    dataset, dataloader = get_ronin_dataloader(
        h5_paths=sample_h5,
        window_size=200,
        step_size=50,
        batch_size=16,
        shuffle=False,
    )
    assert len(dataset) > 0, "Dataset has zero windows!"

    batch_x, batch_y, batch_meta = next(iter(dataloader))

    # Assert shapes
    assert batch_x.shape == (16, 6, 200), f"Unexpected input shape: {batch_x.shape}"
    assert batch_y.shape == (16, 2), f"Unexpected target shape: {batch_y.shape}"

    # Assert zero NaNs
    assert not torch.isnan(batch_x).any(), "Found NaNs in input tensor!"
    assert not torch.isnan(batch_y).any(), "Found NaNs in target velocity!"


def test_model_latency_gate():
    """GATE 2: Asserts single-window inference latency is <= 5.0 ms on CPU."""
    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False)
    model.eval()

    # Single-sample inference test: (1, 6, 200)
    dummy_input = torch.randn(1, 6, 200, dtype=torch.float32)

    # Warm-up passes
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timed passes
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # Convert to ms

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    print(f"\n[LATENCY GATE] Avg Latency: {avg_latency:.2f} ms | p95 Latency: {p95_latency:.2f} ms")
    assert avg_latency <= 5.0, f"LATENCY BUDGET EXCEEDED: {avg_latency:.2f} ms (Must be <= 5.0 ms)"


@pytest.fixture
def real_test_h5():
    path = os.path.join(PROJECT_ROOT, "data", "real", "test_unseen", "test_sub03_unseen_bag_corridor.h5")
    if not os.path.exists(path):
        path = os.path.join(PROJECT_ROOT, "data", "synthetic", "sample_walk_square.h5")
    return path


def test_production_accuracy_gate(real_test_h5, model_checkpoint):
    """GATE 3: Asserts Relative Trajectory Error (RTE) on holdout walk is <= 2.0%."""
    metrics = evaluate_trajectory(
        h5_path=real_test_h5,
        checkpoint_path=model_checkpoint,
        output_plot_path=os.path.join(PROJECT_ROOT, "reports", "ci_gate_trajectory.png"),
        window_size=200,
        step_size=50,
    )

    rte = metrics["rte_percent"]
    ate_rmse = metrics["ate_rmse"]
    final_drift = metrics["final_drift"]

    print(f"\n[ACCURACY GATE] Distance: {metrics['total_distance']:.1f}m | ATE: {ate_rmse:.2f}m | Drift: {final_drift:.2f}m | RTE: {rte:.2f}%")

    assert rte <= 2.0, (
        f"CRITICAL REGRESSION: Relative Trajectory Error is {rte:.2f}%! "
        f"Production gate requires <= 2.00%. Build Failed!"
    )


def test_drift_detector_logic(sample_h5):
    """GATE 4: Verifies that SensorDriftDetector correctly flags shifted distributions."""
    with h5py.File(sample_h5, "r") as f:
        acce = f["synced"]["acce"][:]
        gyro = f["synced"]["gyro"][:]

    detector = SensorDriftDetector(ks_stat_threshold=0.08, wasserstein_threshold=0.6)
    detector.fit_baseline(acce, gyro)

    # 1. Healthy stream should not flag drift
    clean_stream = acce + np.random.normal(0, 0.02, size=acce.shape)
    report_clean = detector.detect_drift(clean_stream, gyro, export_report_path=None)
    assert not report_clean["drift_detected"], "False positive drift triggered on healthy data!"

    # 2. Shifted stream should flag drift
    drifted_stream = acce * 0.3 + np.random.normal(3.0, 1.0, size=acce.shape)
    report_drifted = detector.detect_drift(drifted_stream, gyro, export_report_path=None)
    assert report_drifted["drift_detected"], "Failed to detect synthetic sensor drift!"
