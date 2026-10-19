"""
Dual-Regime Model Training and Threshold Sweep
Trains a single unified model that uses w_vel = 2.1 on straight/gentle paths and w_vel = 2.7 on curved paths.
Sweeps the turn rate threshold across 10 values centered at 0.25 rad/s with step size 0.02 rad/s.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import get_ronin_dataloader, PreprocessedSequence
from scripts.sweep_w_vel import evaluate_sequence

class DualRegimePhysicsLoss(nn.Module):
    """
    Dual-Regime Loss:
    - When turn_rate < threshold (straight / casual walking): applies w_vel = 2.1
    - When turn_rate >= threshold (active curves / Figure-8): applies w_vel = 2.7
    - Fixed direction weight: w_dir_turn_max = 2.5, w_dir_base = 0.5
    """
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

    def forward(
        self,
        v_pred: torch.Tensor,
        v_true: torch.Tensor,
        is_stationary: torch.Tensor,
        turn_rate: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        huber_raw = self.huber_base(v_pred, v_true)
        huber_per_sample = torch.mean(huber_raw, dim=1)

        speed_true = torch.norm(v_true, dim=1)
        moving_mask = speed_true > 0.1

        # Dual-Regime Velocity Weighting
        if turn_rate is not None:
            # Smooth sigmoid transition centered at turn_threshold with width 0.03 rad/s
            # When turn_rate << threshold -> weight = 2.1
            # When turn_rate >> threshold -> weight = 2.7
            gate = torch.sigmoid((turn_rate - self.turn_threshold) / 0.03)
            vel_weights = self.w_vel_straight + (self.w_vel_curve - self.w_vel_straight) * gate
            loss_vel = torch.mean(vel_weights * huber_per_sample)
        else:
            loss_vel = torch.mean(self.w_vel_straight * huber_per_sample)

        # Direction Cosine Loss (scaled on active turns up to 2.5)
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

        # Stationary Penalty
        stat_mask = is_stationary > 0.5
        if stat_mask.sum() > 0:
            speed_pred_sq = torch.sum(v_pred[stat_mask] ** 2, dim=1)
            loss_stat = torch.mean(speed_pred_sq)
        else:
            loss_stat = torch.tensor(0.0, device=v_pred.device)

        total_loss = loss_vel + loss_dir + self.w_stat * loss_stat
        return total_loss, {
            "total": float(total_loss.item()),
            "vel": float(loss_vel.item()),
            "dir": float(loss_dir.item()) if isinstance(loss_dir, torch.Tensor) else float(loss_dir),
        }

def train_dual_regime_model(
    threshold: float,
    train_dir: str,
    save_path: str,
    device: torch.device,
    epochs: int = 3,
):
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
        w_stat=1.0
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

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
            loss, l_dict = loss_fn(vel_pred, batch_y, is_stat, turn_rate=turn_rate)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(l_dict["total"])

        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "loss": best_loss}, save_path)

    return save_path

def run_threshold_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(PROJECT_ROOT, "data/real/train")

    test_seqs = [
        ("Pocket Loop", PreprocessedSequence(os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5"), beta=0.15)),
        ("Handheld Mixed", PreprocessedSequence(os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5"), beta=0.15)),
        ("Bag Corridor", PreprocessedSequence(os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5"), beta=0.15)),
        ("Curved Figure-8", PreprocessedSequence(os.path.join(PROJECT_ROOT, "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5"), beta=0.15)),
    ]

    # 10 Threshold values centered at 0.25 with step size 0.02 rad/s
    thresholds = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35]

    print("=" * 115)
    print("STARTING DUAL-REGIME THRESHOLD SWEEP (10 MODELS CENTERED AT 0.25 rad/s)")
    print("=" * 115)

    all_results = []

    for idx, thresh in enumerate(thresholds):
        tag = f"thresh_{int(thresh * 100):02d}"
        ckpt_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/ronin_dual_regime_{tag}.pt")
        print(f"\n[{idx+1}/{len(thresholds)}] Training Dual-Regime Model with Threshold = {thresh:.2f} rad/s ({thresh * 180 / np.pi:.1f}°/s)...")

        t0 = time.time()
        train_dual_regime_model(
            threshold=thresh,
            train_dir=train_dir,
            save_path=ckpt_path,
            device=device,
            epochs=3
        )
        elapsed = time.time() - t0

        # Evaluate across all 4 test environments
        model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        model.eval()

        row = {"threshold": thresh, "time": elapsed}
        total_drift = 0.0
        total_ate = 0.0

        for name, seq in test_seqs:
            ev = evaluate_sequence(model, seq, device)
            row[f"{name}_drift"] = ev["final_drift"]
            row[f"{name}_ate"] = ev["ate_rmse"]
            row[f"{name}_rte"] = ev["rte_pct"]
            total_drift += ev["final_drift"]
            total_ate += ev["ate_rmse"]

        row["avg_drift"] = total_drift / len(test_seqs)
        row["avg_ate"] = total_ate / len(test_seqs)
        all_results.append(row)

        print(f"  ✓ Done in {elapsed:.1f}s | Figure-8: {row['Curved Figure-8_drift']:.2f}m | Pocket: {row['Pocket Loop_drift']:.2f}m | Avg Drift: {row['avg_drift']:.2f}m")

    # Final summary table
    print("\n" + "=" * 115)
    print("DUAL-REGIME THRESHOLD SWEEP RESULTS")
    print("=" * 115)
    header = f"{'Threshold (rad/s)':<18} | {'Pocket Loop':<14} | {'Handheld':<14} | {'Bag Corridor':<14} | {'Figure-8':<14} | {'Overall Avg Drift':<18} | {'Overall Avg ATE':<16}"
    print(header)
    print("-" * 115)

    best_fig8 = min(all_results, key=lambda x: x["Curved Figure-8_drift"])
    best_overall = min(all_results, key=lambda x: x["avg_drift"])

    for r in all_results:
        t_str = f"{r['threshold']:.2f} rad/s ({r['threshold']*180/np.pi:.1f}°/s)"
        star = " ★ BEST OVERALL" if r["threshold"] == best_overall["threshold"] else (" 🏆 BEST FIG-8" if r["threshold"] == best_fig8["threshold"] else "")
        print(f"{t_str:<18} | {r['Pocket Loop_drift']:>6.2f}m ({r['Pocket Loop_rte']:.2f}%) | {r['Handheld Mixed_drift']:>6.2f}m ({r['Handheld Mixed_rte']:.2f}%) | {r['Bag Corridor_drift']:>6.2f}m ({r['Bag Corridor_rte']:.2f}%) | {r['Curved Figure-8_drift']:>6.2f}m ({r['Curved Figure-8_rte']:.2f}%) | {r['avg_drift']:>9.2f}m        | {r['avg_ate']:>8.2f}m{star}")
    print("=" * 115)

if __name__ == "__main__":
    run_threshold_sweep()

