"""
PyTorch Sliding-Window Dataset for RoNIN Inertial Navigation
Slices continuous 200 Hz IMU sequences into 1.0-second (200-sample) windows,
applies Madgwick attitude estimation and Heading-Independent Frame (HIF) rotation,
and pairs each window with ground-truth 2D velocity targets.
"""

import os
import sys
from typing import List, Tuple, Dict, Optional, Union

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.physics.madgwick import MadgwickAHRS
from src.physics.coordinate_transforms import (
    rotate_body_to_world,
    subtract_gravity,
    extract_yaw_from_quats,
    project_to_heading_independent_frame,
    detect_stationary_periods,
)
from src.data.augmentations import IMUDataAugmenter


def rotate_2d_vector(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotates a 2D vector [x, y] by angle_rad.
    v_rot = [cos(θ)*x - sin(θ)*y, sin(θ)*x + cos(θ)*y]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    if vec.ndim == 1:
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float32)
    else:
        rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
        return vec @ rot_mat.T


class PreprocessedSequence:
    """
    Holds a pre-processed continuous IMU recording in World Frame
    ready for sliding-window segmentation.
    """
    def __init__(
        self,
        h5_path: str,
        beta: float = 0.15,
        use_gt_orientation: bool = False,
    ):
        self.h5_path = h5_path
        self.file_name = os.path.basename(h5_path)

        with h5py.File(h5_path, "r") as f:
            if "synced" in f:
                synced = f["synced"]
            else:
                synced = f

            time_raw = synced["time"][:]
            if time_raw.ndim > 1:
                self.time = time_raw[:, 0].astype(np.float64)
            else:
                self.time = time_raw.astype(np.float64)

            self.acce_body = synced["acce"][:].astype(np.float32)
            self.gyro_body = synced["gyro"][:].astype(np.float32)

            # Ground-truth Position: Check synced/pos, pose/tango_pos, or synced/tango_pos
            if "pos" in synced:
                self.pos_gt = synced["pos"][:].astype(np.float32)
            elif "pose" in f and "tango_pos" in f["pose"]:
                self.pos_gt = f["pose"]["tango_pos"][:].astype(np.float32)
            elif "tango_pos" in synced:
                self.pos_gt = synced["tango_pos"][:].astype(np.float32)
            else:
                raise KeyError(f"Ground truth position not found in {h5_path}")

            # Ground-truth Orientation: Check synced/ori, pose/tango_ori, or synced/game_rv / rv
            if "ori" in synced:
                self.ori_gt = synced["ori"][:].astype(np.float32)
            elif "pose" in f and "tango_ori" in f["pose"]:
                self.ori_gt = f["pose"]["tango_ori"][:].astype(np.float32)
            elif "game_rv" in synced:
                self.ori_gt = synced["game_rv"][:].astype(np.float32)
            elif "rv" in synced:
                self.ori_gt = synced["rv"][:].astype(np.float32)
            else:
                # Default identity quaternion [w, x, y, z]
                self.ori_gt = np.zeros((len(self.time), 4), dtype=np.float32)
                self.ori_gt[:, 0] = 1.0

            # Ground-truth Velocity: Check synced/vel or compute via finite differences of pos_gt
            if "vel" in synced:
                self.vel_gt = synced["vel"][:].astype(np.float32)
            else:
                dt_step = float(self.time[1] - self.time[0]) if len(self.time) > 1 else 0.005
                self.vel_gt = np.gradient(self.pos_gt, dt_step, axis=0).astype(np.float32)

        self.N = len(self.time)
        self.dt = float(self.time[1] - self.time[0]) if self.N > 1 else 0.005

        # 1. Orientation Estimation (Step 1)
        if use_gt_orientation or np.any(self.ori_gt != 0):
            self.quats = self.ori_gt
        else:
            ahrs = MadgwickAHRS(sample_period=self.dt, beta=beta)
            # Initialize with true initial orientation to eliminate initial startup transient
            ahrs.reset(initial_q=self.ori_gt[0])
            self.quats = ahrs.process_sequence(self.gyro_body, self.acce_body)

        # 2. Rotate to World Coordinates (Step 2)
        self.accel_world = rotate_body_to_world(self.acce_body, self.quats)
        self.gyro_world = rotate_body_to_world(self.gyro_body, self.quats)

        # 3. Clean Gravity Subtraction
        self.accel_linear_world = subtract_gravity(self.accel_world, g=9.80665)

        # 4. Extract Compass Yaw for each sample
        self.yaw_angles = extract_yaw_from_quats(self.quats)

        # 5. Stationary Detection (ZUPT)
        self.stationary_mask = detect_stationary_periods(
            self.acce_body, self.gyro_body, window_samples=40, var_threshold=0.08
        )


class RoNINDataset(Dataset):
    """
    PyTorch Dataset providing sliding windows of IMU signals aligned
    to the Heading-Independent Frame (HIF).

    Input Shape:  (6, window_size) -> [a_hif_x, a_hif_y, a_hif_z, w_hif_x, w_hif_y, w_hif_z]
    Target Shape: (2,)             -> [v_hif_x, v_hif_y] (average velocity in HIF)
    """
    def __init__(
        self,
        h5_paths: Union[str, List[str]],
        window_size: int = 200,   # 1.0 second at 200 Hz
        step_size: int = 10,       # 0.05 second step = 20 Hz output rate
        beta: float = 0.15,
        use_gt_orientation: bool = False,
        augment: bool = False,
    ):
        """
        Args:
            h5_paths: Single path or list of paths to RoNIN HDF5 sequence files.
            window_size: Number of time steps per window (default: 200 = 1.0s).
            step_size: Stride between successive windows (default: 10 = 0.05s).
            beta: Madgwick filter gain.
            use_gt_orientation: If True, uses ground-truth orientation (for sanity-checking).
            augment: If True, applies random physical augmentations (yaw rotation, jitter, scaling).
        """
        super().__init__()
        resolved_paths: List[str] = []
        raw_paths = [h5_paths] if isinstance(h5_paths, str) else list(h5_paths)
        for p in raw_paths:
            if os.path.isdir(p):
                for fname in sorted(os.listdir(p)):
                    if fname.endswith((".h5", ".hdf5")):
                        resolved_paths.append(os.path.join(p, fname))
            elif os.path.isfile(p):
                resolved_paths.append(p)
            else:
                raise FileNotFoundError(f"Path not found: {p}")

        if not resolved_paths:
            raise ValueError(f"No .h5 or .hdf5 files found in: {h5_paths}")

        self.h5_paths = resolved_paths
        self.window_size = window_size
        self.step_size = step_size
        self.augment = augment
        self.augmenter = IMUDataAugmenter() if augment else None

        # Preprocess each sequence
        self.sequences: List[PreprocessedSequence] = []
        for path in self.h5_paths:
            seq = PreprocessedSequence(path, beta=beta, use_gt_orientation=use_gt_orientation)
            self.sequences.append(seq)

        # Index mapping: list of (seq_idx, start_idx)
        self.index_map: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            n_samples = seq.N
            for start_idx in range(0, n_samples - window_size + 1, step_size):
                self.index_map.append((seq_idx, start_idx))

        print(
            f"RoNINDataset loaded {len(self.sequences)} sequence(s) | "
            f"Total windows: {len(self.index_map):,} (window={window_size}, step={step_size})"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[float, int]]]:
        seq_idx, start_idx = self.index_map[idx]
        seq = self.sequences[seq_idx]
        end_idx = start_idx + self.window_size

        # 1. Window Slices
        accel_win = seq.accel_linear_world[start_idx:end_idx]  # (window_size, 3)
        gyro_win = seq.gyro_world[start_idx:end_idx]          # (window_size, 3)

        # 2. Heading-Independent Frame (HIF) Alignment
        # Anchor yaw is the heading at the start of the window
        anchor_yaw = float(seq.yaw_angles[start_idx])
        a_hif, w_hif = project_to_heading_independent_frame(accel_win, gyro_win, anchor_yaw)

        # Combine into (6, window_size) tensor for 1D-CNN (Channels, Time_Steps)
        features = np.concatenate([a_hif, w_hif], axis=1)      # (window_size, 6)
        features = np.transpose(features, (1, 0))              # (6, window_size)
        x_tensor = torch.from_numpy(features.astype(np.float32))

        # 3. Ground Truth Target Velocity in HIF
        # The true displacement across the window divided by window duration
        delta_pos = seq.pos_gt[end_idx - 1] - seq.pos_gt[start_idx]  # [dx, dy, dz]
        duration = float(seq.time[end_idx - 1] - seq.time[start_idx])
        v_world_2d = delta_pos[:2] / duration  # [vx, vy] in World Frame

        # Rotate World Velocity into Heading-Independent Frame
        v_hif_2d = rotate_2d_vector(v_world_2d, -anchor_yaw)
        y_tensor = torch.from_numpy(v_hif_2d.astype(np.float32))

        # 4. Extra Metadata (useful for evaluation, trajectory drawing, and multi-task loss)
        stationary_ratio = float(seq.stationary_mask[start_idx:end_idx].mean())
        meta = {
            "seq_idx": seq_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "anchor_yaw": anchor_yaw,
            "time_start": float(seq.time[start_idx]),
            "time_end": float(seq.time[end_idx - 1]),
            "pos_start_x": float(seq.pos_gt[start_idx, 0]),
            "pos_start_y": float(seq.pos_gt[start_idx, 1]),
            "is_stationary": 1.0 if stationary_ratio > 0.6 else 0.0,
        }

        # Apply Data Augmentations (if enabled for training)
        if self.augment and self.augmenter is not None:
            x_tensor, y_tensor = self.augmenter(x_tensor, y_tensor)

        return x_tensor, y_tensor, meta


def get_ronin_dataloader(
    h5_paths: Union[str, List[str]],
    window_size: int = 200,
    step_size: int = 10,
    batch_size: int = 64,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 0,
) -> Tuple[RoNINDataset, DataLoader]:
    """
    Helper function to instantiate RoNINDataset and its DataLoader.
    """
    dataset = RoNINDataset(
        h5_paths=h5_paths,
        window_size=window_size,
        step_size=step_size,
        augment=augment,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
    return dataset, dataloader


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFYING PHASE 3: RoNIN SLIDING-WINDOW DATASET & DATALOADER")
    print("=" * 70)

    test_h5 = os.path.join(PROJECT_ROOT, "data", "synthetic", "sample_walk_square.h5")
    if not os.path.exists(test_h5):
        print(f"Error: test file not found at {test_h5}")
        sys.exit(1)

    # Instantiate dataset with window_size=200 (1s) and step_size=10 (50ms)
    dataset, dataloader = get_ronin_dataloader(
        h5_paths=test_h5,
        window_size=200,
        step_size=10,
        batch_size=32,
        shuffle=True,
    )

    print(f"\n• Total Windows Generated: {len(dataset):,}")
    print(f"• Total Batches (size=32): {len(dataloader):,}")

    # Inspect the first batch
    for batch_idx, (batch_x, batch_y, batch_meta) in enumerate(dataloader):
        print(f"\n--- BATCH #{batch_idx + 1} INSPECTION ---")
        print(f"• Input Tensor X shape : {batch_x.shape} (Expected: [Batch, 6 Channels, 200 Steps])")
        print(f"• Target Tensor Y shape: {batch_y.shape} (Expected: [Batch, 2 Velocity Components])")
        print(f"• Input X dtype / device: {batch_x.dtype} / {batch_x.device}")
        print(f"• Target Y dtype       : {batch_y.dtype}")
        print(f"• Sample Velocity [vx, vy]: {batch_y[0].numpy()} m/s")
        print(f"• Stationary Flags in batch: {batch_meta['is_stationary'][:5].numpy()}")

        # Sanity checks
        assert not torch.isnan(batch_x).any(), "NaN found in input features!"
        assert not torch.isnan(batch_y).any(), "NaN found in target velocity!"
        assert batch_x.shape[1] == 6, f"Expected 6 channels, got {batch_x.shape[1]}"
        assert batch_x.shape[2] == 200, f"Expected 200 time steps, got {batch_x.shape[2]}"
        assert batch_y.shape[1] == 2, f"Expected 2 velocity outputs, got {batch_y.shape[1]}"
        print("\n✓ ALL DATA QUALITY CHECKS PASSED: Zero NaNs, Correct Tensor Shapes!")
        break
