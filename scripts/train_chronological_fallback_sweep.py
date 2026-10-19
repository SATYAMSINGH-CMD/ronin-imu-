"""
Chronological Timeflow Training & Fallback Sweep (1.0s, 1.1s, 1.2s, 1.3s, 1.4s)
Preserves exact chronological timeline (shuffle=False) and passes true continuous turn duration to the loss.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.ronin_resnet import RoNINResNet1D
from src.data.dataset import PreprocessedSequence, rotate_2d_vector, project_to_heading_independent_frame
from scripts.sweep_fallback_time import preextract_sequence, run_inference, score_trajectory

class ChronologicalSequence:
    """Preprocessed sequence with continuous timeline turn duration."""
    def __init__(self, h5_path: str, turn_threshold: float = 0.20):
        self.seq = PreprocessedSequence(h5_path, beta=0.15)
        self.N = self.seq.N
        self.dt = self.seq.dt

        # Continuous Gyroscope Yaw Turn Rate
        gyro_z = np.abs(self.seq.gyro_world[:, 2])
        is_turning = gyro_z >= turn_threshold

        # Calculate continuous turn duration for every millisecond in sequence
        self.turn_durations = np.zeros(self.N, dtype=np.float32)
        cur_dur = 0.0
        for i in range(self.N):
            if is_turning[i]:
                cur_dur += self.dt
                self.turn_durations[i] = cur_dur
            else:
                cur_dur = 0.0
                self.turn_durations[i] = 0.0

class ChronologicalDataset(Dataset):
    """Dataset providing continuous chronological flow with true turn duration metadata."""
    def __init__(self, h5_paths: list, window_size: int = 200, step_size: int = 200, turn_threshold: float = 0.20):
        self.window_size = window_size
        self.step_size = step_size
        self.sequences = [ChronologicalSequence(p, turn_threshold) for p in h5_paths]

        self.index_map = []
        for seq_idx, s in enumerate(self.sequences):
            for start_idx in range(0, s.N - window_size + 1, step_size):
                self.index_map.append((seq_idx, start_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.index_map[idx]
        s = self.sequences[seq_idx]
        end_idx = start_idx + self.window_size

        accel_win = s.seq.accel_linear_world[start_idx:end_idx]
        gyro_win = s.seq.gyro_world[start_idx:end_idx]
        anchor_yaw = float(s.seq.yaw_angles[start_idx])

        a_hif, w_hif = project_to_heading_independent_frame(accel_win, gyro_win, anchor_yaw)
        features = np.concatenate([a_hif, w_hif], axis=1).T
        x_tensor = torch.from_numpy(features.astype(np.float32))

        # Target velocity in HIF
        delta_pos = s.seq.pos_gt[end_idx - 1] - s.seq.pos_gt[start_idx]
        duration = float(s.seq.time[end_idx - 1] - s.seq.time[start_idx])
        v_world_2d = delta_pos[:2] / duration
        v_hif_2d = rotate_2d_vector(v_world_2d, -anchor_yaw)
        y_tensor = torch.from_numpy(v_hif_2d.astype(np.float32))

        # Turn duration reached by this window
        turn_dur = float(s.turn_durations[end_idx - 1])
        stat_ratio = float(s.seq.stationary_mask[start_idx:end_idx].mean())

        meta = {
            "turn_duration": turn_dur,
            "is_stationary": 1.0 if stat_ratio > 0.6 else 0.0,
            "mean_turn_rate": float(np.mean(np.abs(gyro_win[:, 2]))),
        }
        return x_tensor, y_tensor, meta

class ChronoDwellPhysicsLoss(nn.Module):
    def __init__(self, fallback_time_sec: float, turn_threshold: float = 0.20):
        super().__init__()
        self.fallback_time_sec = fallback_time_sec
        self.turn_threshold = turn_threshold
        self.huber_base = nn.HuberLoss(delta=0.5, reduction="none")
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, v_pred, v_true, is_stat, turn_rate, turn_duration):
        huber_raw = self.huber_base(v_pred, v_true)
        huber_per_sample = torch.mean(huber_raw, dim=1)

        speed_true = torch.norm(v_true, dim=1)
        moving_mask = speed_true > 0.1

        # Smooth sigmoid gate modulated by timeline turn duration
        raw_gate = torch.sigmoid((turn_rate - self.turn_threshold) / 0.03)
        is_sustained = (turn_duration >= self.fallback_time_sec).float()
        effective_gate = raw_gate * is_sustained

        vel_weights = 2.1 + (2.7 - 2.1) * effective_gate
        loss_vel = torch.mean(vel_weights * huber_per_sample)

        # Direction loss scaling
        if moving_mask.sum() > 0:
            cos_alignment = self.cos_sim(v_pred[moving_mask], v_true[moving_mask])
            dir_errors = 1.0 - cos_alignment
            active_turn = turn_rate[moving_mask]
            active_ratio = torch.clamp(active_turn / 0.5, 0.0, 1.0)
            adaptive_weights = 0.5 + (2.5 - 0.5) * active_ratio
            loss_dir = torch.mean(adaptive_weights * dir_errors)
        else:
            loss_dir = torch.tensor(0.0, device=v_pred.device)

        # Stationary loss
        stat_mask = is_stat > 0.5
        if stat_mask.sum() > 0:
            loss_stat = torch.mean(torch.sum(v_pred[stat_mask] ** 2, dim=1))
        else:
            loss_stat = torch.tensor(0.0, device=v_pred.device)

        return loss_vel + loss_dir + loss_stat

def train_chrono_model(fb_time: float, dataset, save_path: str, device: torch.device, epochs: int = 2):
    torch.manual_seed(42)
    np.random.seed(42)

    # EXACT CHRONOLOGICAL TIMELINE (shuffle=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = RoNINResNet1D(in_channels=6, base_channels=64, out_dim=2, with_uncertainty=False).to(device)
    loss_fn = ChronoDwellPhysicsLoss(fallback_time_sec=fb_time, turn_threshold=0.20)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, meta in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            is_stat = meta["is_stationary"].to(device)
            turn_rate = meta["mean_turn_rate"].to(device)
            turn_dur = meta["turn_duration"].to(device)

            optimizer.zero_grad()
            vel_pred, _ = model(batch_x)
            loss = loss_fn(vel_pred, batch_y, is_stat, turn_rate, turn_dur)
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
    train_files = [os.path.join(train_dir, f) for f in sorted(os.listdir(train_dir)) if f.endswith(".h5")]

    print("=" * 125)
    print("STEP 1: LOADING CONTINUOUS CHRONOLOGICAL DATASET (shuffle=False)")
    print("=" * 125)
    t0_ds = time.time()
    chrono_ds = ChronologicalDataset(train_files, window_size=200, step_size=200, turn_threshold=0.20)
    print(f"✓ Loaded {len(chrono_ds):,} chronological windows across {len(train_files)} files in {time.time()-t0_ds:.1f}s")

    print("\n" + "=" * 125)
    print("STEP 2: PRE-EXTRACTING 4 HOLDOUT TEST SEQUENCES (~11.3 km)")
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
    print(f"✓ Successfully pre-extracted {len(test_data)} test sequences.")

    print("\n" + "=" * 125)
    print("STEP 3: TRAINING 5 EXPERIMENTAL FALLBACK MODELS [1.0s, 1.1s, 1.2s, 1.3s, 1.4s]")
    print("=" * 125)
    experimental_times = [1.00, 1.10, 1.20, 1.30, 1.40]

    benchmark_records = []
    for idx, fb in enumerate(experimental_times):
        t0 = time.time()
        tag = f"model_chrono_fb_{int(round(fb * 100)):03d}"
        save_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/v3_fallback/{tag}.pt")
        center_mark = " (Center)" if abs(fb - 1.20) < 1e-4 else ""
        print(f"[{idx+1}/5] Training {tag} (Fallback = {fb:.2f}s{center_mark})...", end=" ", flush=True)

        m = train_chrono_model(fb, chrono_ds, save_path, device, epochs=2)
        m.eval()

        drifts = []
        ates = []
        row = {"time": fb, "label": f"{fb:.2f} s{center_mark}"}
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
        benchmark_records.append(row)
        print(f"Done in {time.time()-t0:.1f}s | Avg Drift: {row['avg_drift']:.2f}m | ATE: {row['avg_ate']:.2f}m", flush=True)

    # FINAL COMPARATIVE TABLE
    print("\n" + "=" * 135)
    print("MASTER BENCHMARK: 5 CHRONOLOGICAL FALLBACK MODELS vs TLIO & RONIN")
    print("=" * 135)
    header = f"{'Model (Fallback Time)':<22} | {'Pocket Loop':<18} | {'Handheld Walk':<18} | {'Bag Corridor':<18} | {'Figure-8 Track':<18} | {'Avg Drift':<12} | {'Avg ATE':<10}"
    print(header)
    print("-" * 135)

    for r in benchmark_records:
        c_time = r["label"]
        c_pock = f"{r['Pocket Loop_drift']:>6.2f}m ({r['Pocket Loop_rte']:.2f}%)"
        c_hand = f"{r['Handheld Walk_drift']:>6.2f}m ({r['Handheld Walk_rte']:.2f}%)"
        c_bag  = f"{r['Bag Corridor_drift']:>6.2f}m ({r['Bag Corridor_rte']:.2f}%)"
        c_fig8 = f"{r['Figure-8 Track_drift']:>6.2f}m ({r['Figure-8 Track_rte']:.2f}%)"
        c_avg  = f"{r['avg_drift']:>6.2f} m"
        c_ate  = f"{r['avg_ate']:>5.2f} m"
        print(f"{c_time:<22} | {c_pock:<18} | {c_hand:<18} | {c_bag:<18} | {c_fig8:<18} | {c_avg:<12} | {c_ate:<10}")

    print("-" * 135)
    print(f"{'CathIAS (TLIO)':<22} | {' 8.50m (0.31%)':<18} | {'52.10m (1.62%)':<18} | {'13.80m (0.56%)':<18} | {'49.30m (1.71%)':<18} | {'30.92 m':<12} | {'18.40 m':<10}")
    print(f"{'RoNIN Baseline':<22} | {'11.24m (0.41%)':<18} | {'68.42m (2.13%)':<18} | {'24.15m (0.98%)':<18} | {'180.98m(6.27%)':<18} | {'71.20 m':<12} | {'44.82 m':<10}")
    print("=" * 135)

if __name__ == "__main__":
    main()
