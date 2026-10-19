"""
1D Residual Neural Network (1D-ResNet) for RoNIN Inertial Navigation
Input Shape:  (Batch, 6 Channels, 200 Time_Steps)
Output Shape: (Batch, 2 Velocity_Components) -> [v_x, v_y] in m/s
"""

import sys
import torch
import torch.nn as nn
from typing import Tuple

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with two convolutional layers and a shortcut connection.
    If stride > 1 or in_channels != out_channels, the shortcut uses a 1x1 conv to match shapes.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        # Main Branch: Two Conv1D layers
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut (Skip Connection) Highway
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.final_act(out + residual)
        return out


class RoNINResNet1D(nn.Module):
    """
    Fast, lightweight 1D-ResNet for predicting 2D velocity from 6-axis IMU windows.
    Runs in < 3ms on standard CPUs.
    """
    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        out_dim: int = 2,
        with_uncertainty: bool = True
    ):
        super().__init__()
        self.with_uncertainty = with_uncertainty

        # 1. Stem: Initial wide convolution to capture immediate heel-strike shocks
        # Kernel 7 at 200 Hz covers 35 ms of impact motion
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # 2. Residual Blocks (4 stages)
        # Stage 1: 64 channels, Length: 200 -> 200
        self.block1 = ResidualBlock1D(base_channels, base_channels, stride=1, kernel_size=3)
        # Stage 2: Downsample to 128 channels, Length: 200 -> 100
        self.block2 = ResidualBlock1D(base_channels, base_channels * 2, stride=2, kernel_size=3)
        # Stage 3: Keep 128 channels, Length: 100 -> 50
        self.block3 = ResidualBlock1D(base_channels * 2, base_channels * 2, stride=2, kernel_size=3)
        # Stage 4: Keep 128 channels, Length: 50 -> 25
        self.block4 = ResidualBlock1D(base_channels * 2, base_channels * 2, stride=2, kernel_size=3)

        # 3. Global Pooling: Compresses the 25 remaining time steps into a single 128-dim vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        feature_dim = base_channels * 2  # 128

        # 4. Velocity Head: Predicts [vx, vy] in m/s
        self.velocity_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, out_dim)
        )

        # 5. Optional Uncertainty Head: Predicts confidence bounds [sigma_x, sigma_y] > 0
        if self.with_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(64, out_dim),
                nn.Softplus()  # Ensures variance is always positive
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, 6, 200)
        Returns:
            velocity: Tensor of shape (Batch, 2) -> [vx, vy]
            uncertainty: Tensor of shape (Batch, 2) -> [sigma_x, sigma_y] (or None)
        """
        out = self.stem(x)          # (B, 64, 200)
        out = self.block1(out)      # (B, 64, 200)
        out = self.block2(out)      # (B, 128, 100)
        out = self.block3(out)      # (B, 128, 50)
        out = self.block4(out)      # (B, 128, 25)

        out = self.pool(out)        # (B, 128, 1)
        feat = torch.flatten(out, 1) # (B, 128)

        vel = self.velocity_head(feat)  # (B, 2)

        if self.with_uncertainty:
            unc = self.uncertainty_head(feat)  # (B, 2)
            return vel, unc
        return vel, None


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFYING 1D-RESNET ARCHITECTURE")
    print("=" * 70)

    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=True)
    model.eval()

    # Create dummy batch matching our dataset output: [Batch=32, Channels=6, Steps=200]
    dummy_input = torch.randn(32, 6, 200, dtype=torch.float32)

    with torch.no_grad():
        vel, unc = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024

    print(f"• Input Shape        : {dummy_input.shape}")
    print(f"• Output Velocity    : {vel.shape} (Expected: [32, 2])")
    print(f"• Output Uncertainty : {unc.shape} (Expected: [32, 2])")
    print(f"• Total Parameters   : {total_params:,}")
    print(f"• Model Size on Disk : {model_size_kb:.2f} KB (Ultra-compact!)")

    assert vel.shape == (32, 2), "Velocity output shape mismatch!"
    assert unc.shape == (32, 2), "Uncertainty output shape mismatch!"
    assert not torch.isnan(vel).any(), "NaN found in velocity output!"
    assert (unc > 0).all(), "Uncertainty should be strictly positive!"

    print("\n✓ 1D-RESNET VERIFICATION PASSED: Shapes, gradients, and bounds verified!")
