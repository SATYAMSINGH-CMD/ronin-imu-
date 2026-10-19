"""
Synthetic RoNIN-Compatible IMU Trajectory Generator
Generates realistic 200 Hz IMU data (Accelerometer, Gyroscope) with ground truth
position and orientation for a square walking path with stationary pauses.
"""

import os
import h5py
import numpy as np


def generate_square_walk(
    duration_sec: float = 55.0,
    sampling_rate: float = 200.0,
    side_length_m: float = 10.0,
    walking_speed_mps: float = 1.0,
    noise_accel: float = 0.05,
    noise_gyro: float = 0.008,
    gyro_bias: float = 0.003,
) -> dict:
    """
    Generates a 10m x 10m square walking path:
    - 0 to 5s: Stationary at start (0, 0)
    - 5 to 15s: Walk Leg 1 (+X direction for 10m)
    - 15 to 16s: 90 deg Turn Left (+Y direction)
    - 16 to 26s: Walk Leg 2 (+Y direction for 10m)
    - 26 to 27s: 90 deg Turn Left (-X direction)
    - 27 to 37s: Walk Leg 3 (-X direction for 10m)
    - 37 to 38s: 90 deg Turn Left (-Y direction)
    - 38 to 48s: Walk Leg 4 (-Y direction back to (0,0))
    - 48 to 55s: Stationary pause at finish
    """
    dt = 1.0 / sampling_rate
    num_samples = int(duration_sec * sampling_rate)
    timestamps = np.arange(num_samples) * dt

    pos_gt = np.zeros((num_samples, 3), dtype=np.float32)
    vel_gt = np.zeros((num_samples, 3), dtype=np.float32)
    yaw_gt = np.zeros(num_samples, dtype=np.float32)

    # Constant Earth gravity along World Z axis
    gravity = 9.80665

    # Human gait parameters
    step_freq = 1.8  # Hz (steps per second)
    accel_vertical_amp = 1.8  # m/s² (vertical bounce while walking)
    accel_forward_amp = 0.8  # m/s² (forward push per step)

    current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    current_yaw = 0.0  # radians (0 = pointing +X)

    for i in range(num_samples):
        t = timestamps[i]

        # Phase 0: Stationary pause (0 to 5s)
        if t < 5.0:
            vx, vy = 0.0, 0.0
            omega_z = 0.0

        # Leg 1: Walk +X (5 to 15s)
        elif 5.0 <= t < 15.0:
            current_yaw = 0.0
            vx = walking_speed_mps
            vy = 0.0
            omega_z = 0.0

        # Turn 1: 90 deg turn toward +Y (15 to 16s)
        elif 15.0 <= t < 16.0:
            vx, vy = 0.0, 0.0
            omega_z = (np.pi / 2.0) / 1.0  # +90 deg/s
            current_yaw += omega_z * dt

        # Leg 2: Walk +Y (16 to 26s)
        elif 16.0 <= t < 26.0:
            current_yaw = np.pi / 2.0
            vx = 0.0
            vy = walking_speed_mps
            omega_z = 0.0

        # Turn 2: 90 deg turn toward -X (26 to 27s)
        elif 26.0 <= t < 27.0:
            vx, vy = 0.0, 0.0
            omega_z = (np.pi / 2.0) / 1.0
            current_yaw += omega_z * dt

        # Leg 3: Walk -X (27 to 37s)
        elif 27.0 <= t < 37.0:
            current_yaw = np.pi
            vx = -walking_speed_mps
            vy = 0.0
            omega_z = 0.0

        # Turn 3: 90 deg turn toward -Y (37 to 38s)
        elif 37.0 <= t < 38.0:
            vx, vy = 0.0, 0.0
            omega_z = (np.pi / 2.0) / 1.0
            current_yaw += omega_z * dt

        # Leg 4: Walk -Y (38 to 48s)
        elif 38.0 <= t < 48.0:
            current_yaw = 3.0 * np.pi / 2.0
            vx = 0.0
            vy = -walking_speed_mps
            omega_z = 0.0

        # Finish: Stationary (48 to 55s)
        else:
            vx, vy = 0.0, 0.0
            omega_z = 0.0
            current_yaw = 0.0

        current_pos[0] += vx * dt
        current_pos[1] += vy * dt

        pos_gt[i] = current_pos
        vel_gt[i] = [vx, vy, 0.0]
        yaw_gt[i] = current_yaw

    # Build True Orientation Quaternions [qw, qx, qy, qz]
    # Phone held with a constant 25 degree pitch tilt and dynamic yaw
    pitch_angle = np.radians(25.0)  # Tilt angle towards user's eyes
    roll_angle = np.radians(5.0)    # Slight natural wrist roll

    ori_gt = np.zeros((num_samples, 4), dtype=np.float32)
    accel_body = np.zeros((num_samples, 3), dtype=np.float32)
    gyro_body = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        yaw = yaw_gt[i]
        speed = np.linalg.norm(vel_gt[i, :2])
        is_moving = speed > 0.1

        # Euler to Quaternion (ZYX convention: Yaw, Pitch, Roll)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch_angle * 0.5)
        sp = np.sin(pitch_angle * 0.5)
        cr = np.cos(roll_angle * 0.5)
        sr = np.sin(roll_angle * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        ori_gt[i] = [qw, qx, qy, qz]

        # 3x3 Rotation matrix: Body to World (R)
        # R rotates a vector from sensor frame to world frame
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float64)

        # World linear acceleration: Step impact dynamics when walking
        if is_moving:
            step_phase = 2.0 * np.pi * step_freq * timestamps[i]
            # Vertical bounce
            a_lin_z = accel_vertical_amp * np.sin(step_phase)
            # Forward push along travel direction
            push = accel_forward_amp * np.cos(step_phase)
            a_lin_x = push * np.cos(yaw)
            a_lin_y = push * np.sin(yaw)
        else:
            a_lin_x, a_lin_y, a_lin_z = 0.0, 0.0, 0.0

        # Accelerometer measures: Specific Force = a_linear + Gravity [0, 0, +9.81]
        a_world_total = np.array([a_lin_x, a_lin_y, a_lin_z + gravity], dtype=np.float64)

        # Transform from World to Sensor Body: a_body = R^T * a_world_total
        a_b = R.T @ a_world_total

        # World angular velocity
        if i > 0 and (timestamps[i] - timestamps[i-1]) > 0:
            dyaw = (yaw_gt[i] - yaw_gt[i-1]) / dt
            # wrap around check
            if dyaw > np.pi / dt:
                dyaw -= 2 * np.pi / dt
            elif dyaw < -np.pi / dt:
                dyaw += 2 * np.pi / dt
        else:
            dyaw = 0.0

        omega_world = np.array([0.0, 0.0, dyaw], dtype=np.float64)
        # Transform angular rate to Sensor Body: omega_b = R^T * omega_world
        w_b = R.T @ omega_world

        # Add realistic sensor noise & constant bias
        noise_a = np.random.normal(0, noise_accel, size=3)
        noise_w = np.random.normal(0, noise_gyro, size=3) + gyro_bias

        accel_body[i] = a_b + noise_a
        gyro_body[i] = w_b + noise_w

    return {
        "time": timestamps[:, None],
        "acce": accel_body,
        "gyro": gyro_body,
        "pos": pos_gt,
        "vel": vel_gt,
        "ori": ori_gt,
    }


def save_to_ronin_h5(data_dict: dict, file_path: str):
    """Saves data matching official RoNIN HDF5 structure."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "w") as f:
        synced = f.create_group("synced")
        synced.create_dataset("time", data=data_dict["time"], dtype="float64")
        synced.create_dataset("acce", data=data_dict["acce"], dtype="float32")
        synced.create_dataset("gyro", data=data_dict["gyro"], dtype="float32")
        synced.create_dataset("pos", data=data_dict["pos"], dtype="float32")
        synced.create_dataset("ori", data=data_dict["ori"], dtype="float32")
        # Store velocity as reference ground truth
        synced.create_dataset("vel", data=data_dict["vel"], dtype="float32")
    print(f"Successfully generated RoNIN synthetic file: {file_path}")
    print(f"Total samples: {len(data_dict['time'])} ({len(data_dict['time'])/200.0:.1f} seconds at 200 Hz)")


if __name__ == "__main__":
    out_dir = os.path.join("data", "synthetic")
    file_path = os.path.join(out_dir, "sample_walk_square.h5")
    data = generate_square_walk()
    save_to_ronin_h5(data, file_path)
