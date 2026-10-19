"""
Data Augmentation Engine for RoNIN Inertial Navigation
Generates physically realistic variations of 1-second IMU windows on the fly during training:
1. Random Horizontal Yaw Perturbations
2. Sensor Jitter (Allan Variance Simulation)
3. Step Amplitude Scaling (Shoe / Weight Variation)
4. Gyroscope Bias Drift Injection
"""

import sys
import numpy as np
import torch
from typing import Tuple, Optional

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class RandomYawPerturbation:
    """
    Randomly rotates horizontal acceleration and target velocity by an angle delta_yaw.
    Forces the network to be robust against slight body orientation misalignments.
    """
    def __init__(self, max_angle_deg: float = 30.0, p: float = 0.8):
        self.max_angle_rad = np.radians(max_angle_deg)
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return x, y

        angle = float(np.random.uniform(-self.max_angle_rad, self.max_angle_rad))
        c = float(np.cos(angle))
        s = float(np.sin(angle))

        x_aug = x.clone()
        y_aug = y.clone()

        # Rotate horizontal acceleration (channels 0: ax, 1: ay)
        ax = x[0, :].clone()
        ay = x[1, :].clone()
        x_aug[0, :] = c * ax - s * ay
        x_aug[1, :] = s * ax + c * ay

        # Rotate horizontal angular velocity (channels 3: wx, 4: wy)
        wx = x[3, :].clone()
        wy = x[4, :].clone()
        x_aug[3, :] = c * wx - s * wy
        x_aug[4, :] = s * wx + c * wy

        # Rotate target velocity vector [vx, vy]
        vx = y[0].clone()
        vy = y[1].clone()
        y_aug[0] = c * vx - s * vy
        y_aug[1] = s * vx + c * vy

        return x_aug, y_aug


class SensorNoiseInjection:
    """
    Simulates cheap MEMS sensor white noise and thermal jitter (Allan variance).
    """
    def __init__(self, accel_noise_std: float = 0.04, gyro_noise_std: float = 0.005, p: float = 0.7):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return x, y

        x_aug = x.clone()

        # Accel noise (channels 0, 1, 2) in m/s^2
        accel_noise = torch.randn_like(x[:3, :]) * self.accel_noise_std
        x_aug[:3, :] += accel_noise

        # Gyro noise (channels 3, 4, 5) in rad/s
        gyro_noise = torch.randn_like(x[3:, :]) * self.gyro_noise_std
        x_aug[3:, :] += gyro_noise

        return x_aug, y


class AmplitudeScaling:
    """
    Simulates variations in body weight, stride aggressiveness, and shoe sole cushioning.
    Scales acceleration by factor in [min_scale, max_scale].
    """
    def __init__(self, min_scale: float = 0.90, max_scale: float = 1.10, p: float = 0.6):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return x, y

        scale = float(np.random.uniform(self.min_scale, self.max_scale))
        x_aug = x.clone()
        y_aug = y.clone()

        # Scale acceleration and target velocity proportionally
        x_aug[:3, :] *= scale
        y_aug *= scale

        return x_aug, y_aug


class GyroBiasDriftInjection:
    """
    Injects realistic constant offset to gyroscope axes to simulate uncalibrated sensors.
    """
    def __init__(self, max_bias_rad_s: float = 0.02, p: float = 0.5):
        self.max_bias = max_bias_rad_s
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return x, y

        bias = torch.FloatTensor(3, 1).uniform_(-self.max_bias, self.max_bias)
        x_aug = x.clone()
        x_aug[3:, :] += bias
        return x_aug, y


class IMUDataAugmenter:
    """
    Chains all physically realistic augmentations together with balanced, realistic noise levels.
    """
    def __init__(self):
        self.transforms = [
            RandomYawPerturbation(max_angle_deg=10.0, p=0.5),
            SensorNoiseInjection(accel_noise_std=0.015, gyro_noise_std=0.002, p=0.5),
            AmplitudeScaling(min_scale=0.95, max_scale=1.05, p=0.4),
            GyroBiasDriftInjection(max_bias_rad_s=0.005, p=0.3),
        ]

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFYING IMU DATA AUGMENTATION PIPELINE")
    print("=" * 70)

    augmenter = IMUDataAugmenter()

    dummy_x = torch.ones(6, 200, dtype=torch.float32)
    dummy_y = torch.tensor([1.20, 0.05], dtype=torch.float32)

    aug_x, aug_y = augmenter(dummy_x, dummy_y)

    print(f"• Original Input X Mean : {dummy_x.mean().item():.4f}")
    print(f"• Augmented Input X Mean: {aug_x.mean().item():.4f}")
    print(f"• Original Target Speed : {dummy_y.numpy()}")
    print(f"• Augmented Target Speed: {aug_y.numpy()}")

    assert aug_x.shape == dummy_x.shape, "Augmentation changed input shape!"
    assert aug_y.shape == dummy_y.shape, "Augmentation changed target shape!"
    assert not torch.isnan(aug_x).any(), "NaN found in augmented tensor!"

    print("\n✓ DATA AUGMENTATION TESTS PASSED: Shapes preserved, realistic variations added!")
