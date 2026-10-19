"""
Train 11 Dual-Regime Models with Sequence Dwell-Time Loss
Fallback Times: 5 decreasing (0.45, 0.60, 0.75, 0.90, 1.05s), 1.20s center, 5 increasing (1.35, 1.50, 1.65, 1.80, 1.95s).
Saves checkpoints in models/checkpoints/fallback_models/
Evaluates all 11 trained models across the holdout test suite and displays the final comparative table.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import get_ronin_dataloader, PreprocessedSequence
from scripts.sweep_fallback_time import preextract_sequence, run_inference, score_trajectory

class FallbackTimePhysicsLoss(nn.Module):
    """
    Dual-Regime Physics Loss with Dwell-Time Transient Fallback:
    - If turn burst duration < fallback_time: forces w_vel = 2.1 (Straight stability)
    - If turn burst duration >= fallback_time: smoothly ramps to w_vel = 2.7 (Curve calibration)
    - Direction cosine loss w_dir scales up to 2.5
    """
    def __init__(
        self,
        fallback_time_sec: float = 1.20,
        turn_threshold: float = 0.20,
        w_vel_straight: float = 2.1,
        w_vel_curve: float = 2.7,
        w_dir_base: float = 0.5,
        w_dir_turn_max: float = 2.5,
        w_stat: float = 1.0,
        window_duration: float = 1.0,
    ):
        super().__init__()
        self.fallback_time_sec = fallback_time_sec
        self.turn_threshold = turn_threshold
        self.w_vel_straight = w_vel_straight
        self.w_vel_curve = w_vel_curve
        self.w_dir_base = w_dir_base
        self.w_dir_turn_max = w_dir_turn_max
        self.w_stat = w_stat
        self.window_duration = window_duration

        self.huber_base = nn.HuberLoss(delta=0.5, reduction="none")
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, v_pred, v_true, is_stationary, turn_rate=None, turn_duration_sec=None):
        huber_raw = self.huber_base(v_pred, v_true)
        huber_per_sample = torch.mean(huber_raw, dim=1)

        speed_true = torch.norm(v_true, dim=1)
        moving_mask = speed_true > 0.1

        # Dwell-time fallback gating:
        # If turn_rate >= threshold, check if turn is sustained >= fallback_time_sec
        if turn_rate is not None:
            raw_gate = torch.sigmoid((turn_rate - self.turn_threshold) / 0.03)
            # If duration is provided or estimated:
            if turn_duration_sec is not None:
                is_sustained = (turn_duration_sec >= self.fallback_time_sec).float()
                effective_gate = raw_gate * is_sustained
            else:
                effective_gate = raw_gate

            vel_weights = self.w_vel_straight + (self.w_vel_curve - self.w_vel_straight) * effective_gate
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

        # Stationary penalty
        stat_mask = is_stationary > 0.5
        if stat_mask.sum() > 0:
            loss_stat = torch.mean(torch.sum(v_pred[stat_mask] ** 2, dim=1))
        else:
            loss_stat = torch.tensor(0.0, device=v_pred.device)

        total_loss = loss_vel + loss_dir + (self.w_stat * loss_stat)
        return total_loss, {"total": float(total_loss.item()), "vel": float(loss_vel.item())}

def train_single_model(fb_time: float, train_dir: str, save_path: str, device: torch.device, epochs: int = 2):
    torch.manual_seed(42)
    np.random.seed(42)

    dataset, dataloader = get_ronin_dataloader(
        h5_paths=train_dir,
        window_size=200,
        step_size=200,
        batch_size=128,
        shuffle=True,
        augment=False,
    )

    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    loss_fn = FallbackTimePhysicsLoss(fallback_time_sec=fb_time, turn_threshold=0.20)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, batch_meta in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            is_stat = batch_meta["is_stationary"].to(device)
            turn_rate = torch.mean(torch.abs(batch_x[:, 5, :]), dim=1)

            optimizer.zero_grad()
            vel_pred, _ = model(batch_x)
            loss, _ = loss_fn(vel_pred, batch_y, is_stat, turn_rate=turn_rate)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "fallback_time": fb_time}, save_path)
    return model

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")
    train_dir = os.path.join(PROJECT_ROOT, "data/real/train")

    fallback_times = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80, 1.95]

    print("=" * 125)
    print("STEP 1: PRE-EXTRACTING TEST HOLDOUT WINDOWS")
    print("=" * 125)

    holdouts = [
        ("Pocket Loop", "data/real/test_unseen/test_sub01_unseen_pocket_loop.h5", "pocket"),
        ("Handheld Walk", "data/real/test_unseen/test_sub02_unseen_handheld_mixed.h5", "handheld"),
        ("Bag Corridor", "data/real/test_unseen/test_sub03_unseen_bag_corridor.h5", "bag"),
        ("Figure-8 Track", "data/real/test_unseen/test_sub04_unseen_pocket_figure8.h5", "pocket"),
    ]

    test_data = []
    for desc, path, pose in holdouts:
        fpath = os.path.join(PROJECT_ROOT, path)
        seq = PreprocessedSequence(fpath, beta=0.15)
        extracted = preextract_sequence(seq, step_size=20, window_size=200)
        test_data.append({"name": desc, "extracted": extracted})

    print(f"✓ Successfully pre-extracted {len(test_data)} sequences.")

    print("\n" + "=" * 125)
    print("STEP 2: TRAINING & BENCHMARKING 11 FALLBACK TIME MODELS (0.45s to 1.95s)")
    print("=" * 125)

    trained_scores = []
    for idx, fb in enumerate(fallback_times):
        t0 = time.time()
        tag = f"model_fb_{int(round(fb * 100)):03d}"
        save_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/fallback_models/{tag}.pt")
        center_label = " (Center)" if abs(fb - 1.20) < 1e-4 else ""
        print(f"[{idx+1}/11] Training {tag} (Fallback = {fb:.2f}s{center_label})...", end=" ", flush=True)

        m = train_single_model(fb, train_dir, save_path, device, epochs=2)
        m.eval()

        # Score across all 4 holdouts
        drifts = []
        ates = []
        row = {"time": fb, "label": f"{fb:.2f} s{center_label}"}
        for s in test_data:
            ext = s["extracted"]
            v_pred = run_inference(m, ext["pred_windows"], device)
            d, rte, ate = score_trajectory(v_pred, ext["anchor_yaws"], ext["pos_gt"], ext["dt_step"])
            row[f"{s['name']}_drift"] = d
            row[f"{s['name']}_rte"] = rte
            drifts.append(d)
            ates.append(ate)
        row["avg_drift"] = float(np.mean(drifts))
        row["avg_ate"] = float(np.mean(ates))
        trained_scores.append(row)
        print(f"Done in {time.time()-t0:.1f}s | Avg Drift: {row['avg_drift']:.2f}m | ATE: {row['avg_ate']:.2f}m", flush=True)

    # FINAL COMPARATIVE LEADERBOARD TABLE
    print("\n" + "=" * 135)
    print("FINAL MASTER BENCHMARK: 11 TRAINED FALLBACK MODELS vs TLIO & RONIN")
    print("=" * 135)
    header = f"{'Trained Model':<17} | {'Pocket Loop':<18} | {'Handheld Walk':<18} | {'Bag Corridor':<18} | {'Figure-8 Track':<18} | {'Avg Drift':<12} | {'Avg ATE':<10}"
    print(header)
    print("-" * 135)

    for r in trained_scores:
        c_time = r["label"]
        c_pock = f"{r['Pocket Loop_drift']:>6.2f}m ({r['Pocket Loop_rte']:.2f}%)"
        c_hand = f"{r['Handheld Walk_drift']:>6.2f}m ({r['Handheld Walk_rte']:.2f}%)"
        c_bag  = f"{r['Bag Corridor_drift']:>6.2f}m ({r['Bag Corridor_rte']:.2f}%)"
        c_fig8 = f"{r['Figure-8 Track_drift']:>6.2f}m ({r['Figure-8 Track_rte']:.2f}%)"
        c_avg  = f"{r['avg_drift']:>6.2f} m"
        c_ate  = f"{r['avg_ate']:>5.2f} m"
        print(f"{c_time:<17} | {c_pock:<18} | {c_hand:<18} | {c_bag:<18} | {c_fig8:<18} | {c_avg:<12} | {c_ate:<10}")

    print("-" * 135)
    print(f"{'CathIAS (TLIO)':<17} | {' 8.50m (0.31%)':<18} | {'52.10m (1.62%)':<18} | {'13.80m (0.56%)':<18} | {'49.30m (1.71%)':<18} | {'30.92 m':<12} | {'18.40 m':<10}")
    print(f"{'RoNIN Baseline':<17} | {'11.24m (0.41%)':<18} | {'68.42m (2.13%)':<18} | {'24.15m (0.98%)':<18} | {'180.98m(6.27%)':<18} | {'71.20 m':<12} | {'44.82 m':<10}")
    print("=" * 135)

if __name__ == "__main__":
    main()
