"""
Training Pipeline for RoNIN 1D-ResNet
Trains the velocity estimation model using multi-task physics loss:
1. Velocity Huber Loss
2. Direction Cosine Loss
3. Stationary ZUPT Penalty
"""

import os
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import get_ronin_dataloader


class PhysicsInformedLoss(nn.Module):
    """
    Composite Loss Function for Inertial Navigation:
    Total Loss = Loss_vel + (w_dir_effective * Loss_dir) + (w_stat * Loss_stationary)
    Supports Idea B with Implementation 1 (Turn-Velocity Balancing):
    - w_dir_base on straight walking (e.g. 0.5)
    - Dynamically scales direction weight up to w_dir_turn_max (e.g. 2.5) during active turns!
    - Dynamically scales velocity weight up to w_vel_turn_max (e.g. 2.0) during active turns to prevent stride shrinkage / baby steps!
    """
    def __init__(
        self,
        w_dir_base: float = 0.5,
        w_dir_turn_max: float = 2.5,
        w_vel_turn_max: float = 2.0,
        w_stat: float = 1.0,
        turn_threshold: float = 0.5,
        huber_delta: float = 0.5
    ):
        super().__init__()
        self.w_dir_base = w_dir_base
        self.w_dir_turn_max = w_dir_turn_max
        self.w_vel_turn_max = w_vel_turn_max
        self.w_stat = w_stat
        self.turn_threshold = turn_threshold
        self.huber_base = nn.HuberLoss(delta=huber_delta, reduction="none")
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(
        self,
        v_pred: torch.Tensor,
        v_true: torch.Tensor,
        is_stationary: torch.Tensor,
        turn_rate: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        # 1. Implementation 1: Turn-Velocity Balanced Huber Loss
        # Prevents the neural network from shrinking velocity magnitude during turns
        huber_raw = self.huber_base(v_pred, v_true)  # (Batch, 2)
        huber_per_sample = torch.mean(huber_raw, dim=1)  # (Batch,)

        speed_true = torch.norm(v_true, dim=1)
        moving_mask = speed_true > 0.1

        if turn_rate is not None and self.w_vel_turn_max > 1.0:
            turn_ratio = torch.clamp(turn_rate / self.turn_threshold, 0.0, 1.0)
            vel_weights = 1.0 + (self.w_vel_turn_max - 1.0) * turn_ratio
            loss_vel = torch.mean(vel_weights * huber_per_sample)
        else:
            loss_vel = torch.mean(huber_per_sample)

        # 2. Direction Cosine Loss (scaled on active turns)
        if moving_mask.sum() > 0:
            cos_alignment = self.cos_sim(v_pred[moving_mask], v_true[moving_mask])
            dir_errors = 1.0 - cos_alignment

            # Idea B: Dynamically scale penalty from w_dir_base (straight) up to w_dir_turn_max (turn)
            if turn_rate is not None and self.w_dir_turn_max > self.w_dir_base:
                active_turn = turn_rate[moving_mask]
                active_ratio = torch.clamp(active_turn / self.turn_threshold, 0.0, 1.0)
                adaptive_weights = self.w_dir_base + (self.w_dir_turn_max - self.w_dir_base) * active_ratio
                loss_dir = torch.mean(adaptive_weights * dir_errors)
            else:
                loss_dir = self.w_dir_base * torch.mean(dir_errors)
        else:
            loss_dir = torch.tensor(0.0, device=v_pred.device)

        # 3. Stationary Penalty (when ZUPT says stationary, heavily penalize any predicted speed)
        stat_mask = is_stationary > 0.5
        if stat_mask.sum() > 0:
            speed_pred_sq = torch.sum(v_pred[stat_mask] ** 2, dim=1)
            loss_stat = torch.mean(speed_pred_sq)
        else:
            loss_stat = torch.tensor(0.0, device=v_pred.device)

        total_loss = loss_vel + loss_dir + self.w_stat * loss_stat

        loss_dict = {
            "total": float(total_loss.item()),
            "vel": float(loss_vel.item()),
            "dir": float(loss_dir.item()) if isinstance(loss_dir, torch.Tensor) else float(loss_dir),
            "stat": float(loss_stat.item()) if isinstance(loss_stat, torch.Tensor) else float(loss_stat),
        }
        return total_loss, loss_dict


def train_model(
    h5_path: str,
    epochs: int = 15,
    batch_size: int = 64,
    step_size: int = 100,
    lr: float = 1e-3,
    w_dir_base: float = 0.5,
    w_dir_turn_max: float = 2.5,
    w_vel_turn_max: float = 2.0,
    augment: bool = False,
    save_path: str = "models/checkpoints/ronin_resnet_best.pt"
):
    print("=" * 70)
    print("RoNIN 1D-RESNET TRAINING PIPELINE")
    print("=" * 70)

    # Set deterministic seeds
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Training Device    : {device}")
    print(f"• Dataset Source     : {h5_path}")
    print(f"• Training Epochs    : {epochs}")
    print(f"• Batch Size         : {batch_size}")
    print(f"• Window Step Size   : {step_size} samples ({step_size * 0.005:.2f}s stride)")
    print(f"• Learning Rate      : {lr}")
    print(f"• Straight Weight    : w_dir_base = {w_dir_base}")
    print(f"• Turn Direction Max : w_dir_turn_max = {w_dir_turn_max}")
    print(f"• Turn Velocity Max  : w_vel_turn_max = {w_vel_turn_max} (Implementation 1)")
    print(f"• Data Augmentation  : {'ENABLED' if augment else 'DISABLED (Pure Real Data)'}")

    # 1. Dataset & DataLoader
    dataset, dataloader = get_ronin_dataloader(
        h5_paths=h5_path,
        window_size=200,
        step_size=step_size,
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
    )
    print(f"• Total Windows      : {len(dataset):,}")
    print(f"• Batches per Epoch  : {len(dataloader):,}")

    # 2. Instantiate Model
    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    loss_fn = PhysicsInformedLoss(
        w_dir_base=w_dir_base,
        w_dir_turn_max=w_dir_turn_max,
        w_vel_turn_max=w_vel_turn_max,
        w_stat=1.0
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    print("\n" + "-" * 70)
    print(f"{'Epoch':<8}{'Total Loss':<14}{'Vel Loss':<14}{'Dir Loss':<14}{'Time':<10}")
    print("-" * 70)

    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_vel_losses = []
        epoch_dir_losses = []
        t0 = time.time()

        for batch_x, batch_y, batch_meta in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            is_stat = batch_meta["is_stationary"].to(device)

            # Idea B: Extract mean yaw turn rate from channel 5 (wz in HIF)
            turn_rate = torch.mean(torch.abs(batch_x[:, 5, :]), dim=1)

            optimizer.zero_grad()
            vel_pred, _ = model(batch_x)
            loss, loss_dict = loss_fn(vel_pred, batch_y, is_stat, turn_rate=turn_rate)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total"])
            epoch_vel_losses.append(loss_dict["vel"])
            epoch_dir_losses.append(loss_dict["dir"])

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_vel = sum(epoch_vel_losses) / len(epoch_vel_losses)
        avg_dir = sum(epoch_dir_losses) / len(epoch_dir_losses)

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                save_path
            )

        print(f"{epoch:<8}{avg_loss:<14.4f}{avg_vel:<14.4f}{avg_dir:<14.4f}{elapsed:<10.2f}s")

    total_time = time.time() - start_all
    print("-" * 70)
    print(f"✓ Training Complete in {total_time:.2f}s! Best Loss: {best_loss:.4f}")
    print(f"✓ Model weights saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RoNIN 1D-ResNet Model Trainer")
    parser.add_argument("--data", type=str, default="data/real/train", help="Path to HDF5 file or directory of sequences")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--step-size", type=int, default=200, help="Sliding window stride in samples")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--w-dir-base", type=float, default=0.5, help="Base direction loss weight on straight walking (default: 0.5)")
    parser.add_argument("--w-dir-turn-max", type=float, default=2.5, help="Idea B: peak direction loss weight on active turns (e.g. 1.0, 1.5, 2.5)")
    parser.add_argument("--w-vel-turn-max", type=float, default=2.0, help="Implementation 1: peak velocity loss weight on active turns (default: 2.0)")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (disabled by default)")
    parser.add_argument("--save-path", type=str, default="models/checkpoints/ronin_resnet_best.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    ckpt_path = os.path.join(PROJECT_ROOT, args.save_path) if not os.path.isabs(args.save_path) else args.save_path

    train_model(
        h5_path=train_data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        step_size=args.step_size,
        lr=args.lr,
        w_dir_base=args.w_dir_base,
        w_dir_turn_max=args.w_dir_turn_max,
        w_vel_turn_max=args.w_vel_turn_max,
        augment=args.augment,
        save_path=ckpt_path
    )
