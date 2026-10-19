"""
Train Dual-Regime Models (thresholds 0.23, 0.27, 0.35) on all 9 training files
Saves checkpoints to models/checkpoints/real_v3/
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import get_ronin_dataloader


class DualRegimePhysicsLoss(nn.Module):
    def __init__(
        self,
        turn_threshold: float = 0.25,
        w_vel_straight: float = 2.1,
        w_vel_curve: float = 2.7,
        w_dir_base: float = 0.5,
        w_dir_turn_max: float = 2.5,
        w_stat: float = 1.0,
    ):
        super().__init__()
        self.turn_threshold = turn_threshold
        self.w_vel_straight = w_vel_straight
        self.w_vel_curve = w_vel_curve
        self.w_dir_base = w_dir_base
        self.w_dir_turn_max = w_dir_turn_max
        self.w_stat = w_stat

        self.huber_base = nn.HuberLoss(delta=0.5, reduction="none")
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, v_pred, v_true, is_stationary, turn_rate=None):
        huber_raw = self.huber_base(v_pred, v_true)
        huber_per_sample = torch.mean(huber_raw, dim=1)
        speed_true = torch.norm(v_true, dim=1)
        moving_mask = speed_true > 0.1

        if turn_rate is not None:
            gate = torch.sigmoid((turn_rate - self.turn_threshold) / 0.03)
            vel_weights = self.w_vel_straight + (self.w_vel_curve - self.w_vel_straight) * gate
            loss_vel = torch.mean(vel_weights * huber_per_sample)
        else:
            loss_vel = torch.mean(self.w_vel_straight * huber_per_sample)

        if moving_mask.sum() > 0:
            cos_alignment = self.cos_sim(v_pred[moving_mask], v_true[moving_mask])
            dir_errors = 1.0 - cos_alignment
            if turn_rate is not None and self.w_dir_turn_max > self.w_dir_base:
                active_turn = turn_rate[moving_mask]
                active_ratio = torch.clamp(active_turn / 0.5, 0.0, 1.0)
                adaptive_weights = self.w_dir_base + (self.w_dir_turn_max - self.w_dir_base) * active_ratio
                loss_dir = torch.mean(adaptive_weights * dir_errors)
            else:
                loss_dir = self.w_dir_base * torch.mean(dir_errors)
        else:
            loss_dir = torch.tensor(0.0, device=v_pred.device)

        stat_mask = is_stationary > 0.5
        if stat_mask.sum() > 0:
            speed_pred_sq = torch.sum(v_pred[stat_mask] ** 2, dim=1)
            loss_stat = torch.mean(speed_pred_sq)
        else:
            loss_stat = torch.tensor(0.0, device=v_pred.device)

        return loss_vel + loss_dir + self.w_stat * loss_stat


def train_model(threshold: float, train_dir: str, save_path: str, device: torch.device, epochs: int = 3):
    torch.manual_seed(42)
    np.random.seed(42)

    dataset, dataloader = get_ronin_dataloader(
        h5_paths=train_dir,
        window_size=200,
        step_size=200,
        batch_size=64,
        shuffle=True,
        augment=False,
    )

    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    loss_fn = DualRegimePhysicsLoss(
        turn_threshold=threshold,
        w_vel_straight=2.1,
        w_vel_curve=2.7,
        w_dir_base=0.5,
        w_dir_turn_max=2.5,
        w_stat=1.0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_meta in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            is_stat = batch_meta["is_stationary"].to(device)
            turn_rate = torch.mean(torch.abs(batch_x[:, 5, :]), dim=1)

            optimizer.zero_grad()
            vel_pred, _ = model(batch_x)
            loss = loss_fn(vel_pred, batch_y, is_stat, turn_rate=turn_rate)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "loss": best_loss, "threshold": threshold}, save_path)

    elapsed = time.time() - t_start
    print(f"✓ Trained Dual-Regime (threshold={threshold:.2f} rad/s) in {elapsed:.1f}s | Best Loss: {best_loss:.6f} -> {save_path}")
    return save_path


def main():
    device = torch.device("cpu")
    train_dir = os.path.join(PROJECT_ROOT, "data/real/train")

    print("=" * 100)
    print("TRAINING DUAL-REGIME MODELS ON EXPANDED 9-FILE REAL DATASET")
    print(f"Train directory: {train_dir}")
    print("=" * 100)

    for th in [0.23, 0.27, 0.35]:
        save_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/real_v3/model_dual_regime_thresh_{int(round(th*100)):02d}.pt")
        print(f"\nTraining model for threshold = {th:.2f} rad/s...")
        train_model(threshold=th, train_dir=train_dir, save_path=save_path, device=device, epochs=3)

    print("\n" + "=" * 100)
    print("ALL DUAL-REGIME MODELS TRAINED SUCCESSFULLY ON 9-FILE DATASET!")
    print("=" * 100)


if __name__ == "__main__":
    main()
