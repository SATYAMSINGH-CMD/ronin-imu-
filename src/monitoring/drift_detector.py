"""
Production Sensor Data Drift Detector for Inertial Navigation
Monitors incoming live IMU streams (accelerometer and gyroscope) against
the training baseline distribution using Kolmogorov-Smirnov (KS) tests
and Wasserstein distance metrics.
Generates automated drift reports: reports/drift_report.json
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
from scipy import stats

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class SensorDriftDetector:
    """
    Detects sensor hardware bias, vibration dampening, or behavioral distribution shifts.
    """
    def __init__(
        self,
        ks_stat_threshold: float = 0.08,
        wasserstein_threshold: float = 0.6
    ):
        self.ks_stat_threshold = ks_stat_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.baseline_stats: Optional[Dict[str, Any]] = None
        self.baseline_accel_mag: Optional[np.ndarray] = None
        self.baseline_gyro_mag: Optional[np.ndarray] = None

    def fit_baseline(self, accel: np.ndarray, gyro: np.ndarray) -> Dict[str, Any]:
        """
        Computes reference baseline statistical profile from training dataset.
        Args:
            accel: Shape (N, 3) in m/s^2
            gyro: Shape (N, 3) in rad/s
        """
        accel_mag = np.linalg.norm(accel, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        # Subsample if dataset is very large to speed up KS test (keep up to 10,000 points)
        if len(accel_mag) > 10000:
            indices = np.random.choice(len(accel_mag), 10000, replace=False)
            self.baseline_accel_mag = accel_mag[indices]
            self.baseline_gyro_mag = gyro_mag[indices]
        else:
            self.baseline_accel_mag = accel_mag
            self.baseline_gyro_mag = gyro_mag

        self.baseline_stats = {
            "samples": int(len(accel)),
            "accel": {
                "mean": float(np.mean(accel_mag)),
                "std": float(np.std(accel_mag)),
                "p95": float(np.percentile(accel_mag, 95)),
                "variance": float(np.var(accel, axis=0).sum()),
            },
            "gyro": {
                "mean": float(np.mean(gyro_mag)),
                "std": float(np.std(gyro_mag)),
                "p95": float(np.percentile(gyro_mag, 95)),
                "variance": float(np.var(gyro, axis=0).sum()),
            }
        }
        return self.baseline_stats

    def detect_drift(
        self,
        live_accel: np.ndarray,
        live_gyro: np.ndarray,
        export_report_path: Optional[str] = "reports/drift_report.json"
    ) -> Dict[str, Any]:
        """
        Tests live incoming IMU stream against training baseline.
        Returns:
            Dict containing statistical tests, drift flags, and human-readable diagnosis.
        """
        if self.baseline_accel_mag is None or self.baseline_gyro_mag is None:
            raise ValueError("Baseline has not been fitted! Call fit_baseline() first.")

        live_accel_mag = np.linalg.norm(live_accel, axis=1)
        live_gyro_mag = np.linalg.norm(live_gyro, axis=1)

        # 1. Kolmogorov-Smirnov 2-sample test
        ks_accel = stats.ks_2samp(self.baseline_accel_mag, live_accel_mag)
        ks_gyro = stats.ks_2samp(self.baseline_gyro_mag, live_gyro_mag)

        # 2. Wasserstein distance (Earth Mover's Distance)
        wd_accel = float(stats.wasserstein_distance(self.baseline_accel_mag, live_accel_mag))
        wd_gyro = float(stats.wasserstein_distance(self.baseline_gyro_mag, live_gyro_mag))

        # Check drift conditions using Wasserstein Distance (effect size) and KS Statistic
        accel_drift = bool(wd_accel > self.wasserstein_threshold or ks_accel.statistic > self.ks_stat_threshold)
        gyro_drift = bool(wd_gyro > (self.wasserstein_threshold * 0.5) or ks_gyro.statistic > self.ks_stat_threshold)

        drift_detected = bool(accel_drift or gyro_drift)

        # Severity Diagnosis
        if not drift_detected:
            diagnosis = "HEALTHY: Live sensor distribution matches training baseline."
            severity = "NONE"
        elif accel_drift and gyro_drift:
            diagnosis = "CRITICAL DRIFT: Both accelerometer and gyroscope have shifted significantly. New sensor hardware or novel carrying pose detected!"
            severity = "HIGH"
        else:
            diagnosis = "WARNING: Partial sensor drift detected. Potential temperature bias or altered motion cadence."
            severity = "MEDIUM"

        report = {
            "drift_detected": drift_detected,
            "severity": severity,
            "diagnosis": diagnosis,
            "metrics": {
                "accelerometer": {
                    "wasserstein_distance": round(wd_accel, 4),
                    "ks_statistic": round(float(ks_accel.statistic), 4),
                    "ks_pvalue": float(ks_accel.pvalue),
                    "drift_flag": accel_drift,
                    "live_mean": round(float(np.mean(live_accel_mag)), 3),
                    "baseline_mean": round(self.baseline_stats["accel"]["mean"], 3),
                },
                "gyroscope": {
                    "wasserstein_distance": round(wd_gyro, 4),
                    "ks_statistic": round(float(ks_gyro.statistic), 4),
                    "ks_pvalue": float(ks_gyro.pvalue),
                    "drift_flag": gyro_drift,
                    "live_mean": round(float(np.mean(live_gyro_mag)), 3),
                    "baseline_mean": round(self.baseline_stats["gyro"]["mean"], 3),
                }
            }
        }

        if export_report_path:
            os.makedirs(os.path.dirname(export_report_path), exist_ok=True)
            with open(export_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    import h5py
    print("=" * 70)
    print("TESTING SENSOR DATA DRIFT DETECTOR")
    print("=" * 70)

    sample_h5 = os.path.join(PROJECT_ROOT, "data", "synthetic", "sample_walk_square.h5")
    with h5py.File(sample_h5, "r") as f:
        acce_clean = f["synced"]["acce"][:]
        gyro_clean = f["synced"]["gyro"][:]

    detector = SensorDriftDetector(ks_stat_threshold=0.08, wasserstein_threshold=0.6)
    detector.fit_baseline(acce_clean, gyro_clean)
    print("• Baseline fitted on clean training dataset.")

    # Test 1: Ingesting similar/clean stream (Should NOT trigger drift)
    print("\n--- TEST CASE 1: HEALTHY LIVE SENSOR STREAM ---")
    noisy_clean = acce_clean + np.random.normal(0, 0.05, size=acce_clean.shape)
    report_clean = detector.detect_drift(noisy_clean, gyro_clean)
    print(f"• Drift Detected : {report_clean['drift_detected']}")
    print(f"• Severity       : {report_clean['severity']}")
    print(f"• Diagnosis      : {report_clean['diagnosis']}")
    assert not report_clean["drift_detected"], "Healthy stream triggered false drift alert!"

    # Test 2: Ingesting heavily shifted stream (e.g., loose pocket vibration or hardware bias)
    print("\n--- TEST CASE 2: HEAVILY DRIFTED LIVE STREAM ---")
    drifted_acce = acce_clean * 0.4 + np.random.normal(2.5, 0.8, size=acce_clean.shape)
    drifted_gyro = gyro_clean + 1.2
    report_drift = detector.detect_drift(drifted_acce, drifted_gyro)
    print(f"• Drift Detected : {report_drift['drift_detected']}")
    print(f"• Severity       : {report_drift['severity']}")
    print(f"• Diagnosis      : {report_drift['diagnosis']}")
    print(f"• Accel Wasserstein Dist: {report_drift['metrics']['accelerometer']['wasserstein_distance']}")
    assert report_drift["drift_detected"], "Failed to detect heavily drifted sensor data!"

    print("\n✓ ALL SENSOR DRIFT DETECTOR TESTS PASSED!")
