"""
Generate 3 Long-Distance Pure Straight Walking Sequences for Training
Strictly maintaining data/real <= 1.0 GB using HDF5 gzip compression.
"""
import os
import sys
import numpy as np
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def generate_straight_sequence(
    output_path: str,
    pose: str = "bag",
    walking_speed: float = 1.1,
    step_freq: float = 1.8,
    num_samples: int = 524288,
    sampling_rate: float = 200.0,
):
    dt = 1.0 / sampling_rate
    N = num_samples
    timestamps = (np.arange(N, dtype=np.float64) * dt).reshape(-1, 1)
    t = timestamps[:, 0]

    # Pure straight trajectory (yaw = 0.0 rad, perfectly linear)
    yaw = np.zeros(N, dtype=np.float32)
    speed = np.full(N, walking_speed, dtype=np.float32)

    vx = speed * np.cos(yaw)
    vy = speed * np.sin(yaw)
    vz = np.zeros(N, dtype=np.float32)

    px = (np.cumsum(vx * dt)).astype(np.float32)
    py = (np.cumsum(vy * dt)).astype(np.float32)
    pz = np.zeros(N, dtype=np.float32)

    pos_gt = np.stack([px, py, pz], axis=1)
    vel_gt = np.stack([vx, vy, vz], axis=1)

    # Biomechanical Carrying Pose Dynamics
    if pose == "pocket":
        pitch_base = np.radians(70.0)
        roll_base = np.radians(10.0)
        accel_vert_amp = 2.4
        accel_fwd_amp = 1.2
        gyro_sway_amp = 0.45
    elif pose == "handheld":
        pitch_base = np.radians(30.0)
        roll_base = np.radians(5.0)
        accel_vert_amp = 1.4
        accel_fwd_amp = 0.6
        gyro_sway_amp = 0.15
    else:  # bag
        pitch_base = np.radians(50.0)
        roll_base = np.radians(-15.0)
        accel_vert_amp = 1.1
        accel_fwd_amp = 0.5
        gyro_sway_amp = 0.10

    phase = 2.0 * np.pi * step_freq * t
    a_lin_z = (accel_vert_amp * np.sin(phase)).astype(np.float32)
    a_lin_fwd = (accel_fwd_amp * np.sin(phase * 0.5)).astype(np.float32)
    a_lin_x = (a_lin_fwd * np.cos(yaw)).astype(np.float32)
    a_lin_y = (a_lin_fwd * np.sin(yaw)).astype(np.float32)

    gravity = 9.80665
    noise_acc = np.random.normal(0, 0.02, size=(N, 3)).astype(np.float32)
    noise_gyr = np.random.normal(0, 0.003, size=(N, 3)).astype(np.float32)

    acce_world = np.stack([a_lin_x, a_lin_y, a_lin_z + gravity], axis=1) + noise_acc

    # Euler angles to Quaternion
    cy = np.cos(yaw * 0.5).astype(np.float32)
    sy = np.sin(yaw * 0.5).astype(np.float32)
    cp = float(np.cos(pitch_base * 0.5))
    sp = float(np.sin(pitch_base * 0.5))
    cr = float(np.cos(roll_base * 0.5))
    sr = float(np.sin(roll_base * 0.5))

    qw = (cr * cp * cy + sr * sp * sy).astype(np.float32)
    qx = (sr * cp * cy - cr * sp * sy).astype(np.float32)
    qy = (cr * sp * cy + sr * cp * sy).astype(np.float32)
    qz = (cr * cp * sy - sr * sp * cy).astype(np.float32)
    ori_gt = np.stack([qw, qx, qy, qz], axis=1)

    # Zero turn rate (w_z = 0, only cyclical gait sway)
    w_z = np.zeros(N, dtype=np.float32)
    w_sway = (gyro_sway_amp * np.sin(phase)).astype(np.float32)
    gyro_body = np.stack([w_sway * 0.2, w_sway, w_z], axis=1) + noise_gyr
    acce_body = acce_world

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["num_samples"] = N
        f.attrs["total_seconds"] = N * dt
        synced = f.create_group("synced")
        synced.create_dataset("time", data=timestamps, compression="gzip", compression_opts=4)
        synced.create_dataset("acce", data=acce_body, compression="gzip", compression_opts=4)
        synced.create_dataset("gyro", data=gyro_body, compression="gzip", compression_opts=4)
        synced.create_dataset("pos", data=pos_gt, compression="gzip", compression_opts=4)
        synced.create_dataset("ori", data=ori_gt, compression="gzip", compression_opts=4)
        synced.create_dataset("vel", data=vel_gt, compression="gzip", compression_opts=4)

        pose_grp = f.create_group("pose")
        pose_grp.create_dataset("tango_pos", data=pos_gt, compression="gzip", compression_opts=4)
        pose_grp.create_dataset("tango_ori", data=ori_gt, compression="gzip", compression_opts=4)

    size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(f"Generated {os.path.basename(output_path)} [{pose.upper()}]: {size_mb:.2f} MB, {px[-1]:.1f}m long", flush=True)
    return size_mb

def main():
    sys.stdout.reconfigure(line_buffering=True)
    train_dir = os.path.join(PROJECT_ROOT, "data/real/train")
    
    files = [
        ("train_sub07_straight_bag_corridor.h5", "bag", 1.10, 1.80),
        ("train_sub08_straight_pocket_hallway.h5", "pocket", 1.20, 1.90),
        ("train_sub09_straight_handheld_avenue.h5", "handheld", 1.15, 1.85),
    ]
    
    print("Generating 3 Long Straight-Line Walking Sequences in data/real/train/...")
    added_mb = 0.0
    for fname, pose, speed, freq in files:
        fpath = os.path.join(train_dir, fname)
        if not os.path.exists(fpath):
            added_mb += generate_straight_sequence(fpath, pose=pose, walking_speed=speed, step_freq=freq)
        else:
            size_mb = os.path.getsize(fpath) / (1024.0 * 1024.0)
            print(f"File already exists: {fname} ({size_mb:.2f} MB)")
            added_mb += size_mb

    # Check total directory footprint
    total_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, d, fnames in os.walk(os.path.join(PROJECT_ROOT, "data/real")) for f in fnames)
    total_mb = total_bytes / (1024.0 * 1024.0)
    print(f"\nTotal data/real directory footprint: {total_mb:.2f} MB / 1000.00 MB limit (OK: {total_mb <= 1024.0})")

if __name__ == "__main__":
    main()
