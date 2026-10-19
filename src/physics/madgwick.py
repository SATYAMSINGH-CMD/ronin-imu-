"""
Madgwick AHRS Filter (Attitude and Heading Reference System)
Estimates 3D orientation (quaternion) from 6-axis IMU (Accelerometer + Gyroscope)
using gradient descent on the Earth gravity direction.
"""

import numpy as np


class MadgwickAHRS:
    def __init__(self, sample_period: float = 1.0 / 200.0, beta: float = 0.1):
        """
        Args:
            sample_period: Sampling interval in seconds (e.g. 1/200s = 0.005s)
            beta: Filter gain. Larger beta = faster convergence to gravity vector,
                  Smaller beta = more trust in gyro integration (smoother).
        """
        self.dt = sample_period
        self.beta = beta
        # Initial orientation quaternion: [qw, qx, qy, qz]
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self, initial_q: np.ndarray = None):
        if initial_q is not None:
            self.q = np.array(initial_q, dtype=np.float64)
            self.q /= np.linalg.norm(self.q)
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update_imu(self, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """
        Step-by-step update for 6-axis IMU.
        Args:
            gyro: Angular velocity in rad/s, shape (3,) -> [wx, wy, wz]
            accel: Specific force in m/s², shape (3,) -> [ax, ay, az]
        Returns:
            Current quaternion [qw, qx, qy, qz], shape (4,)
        """
        q = self.q
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        # Rate of change of quaternion from gyroscope: (1/2) * q ⊗ [0, wx, wy, wz]
        gx, gy, gz = gyro[0], gyro[1], gyro[2]
        q_dot_omega = 0.5 * np.array([
            -qx * gx - qy * gy - qz * gz,
             qw * gx + qy * gz - qz * gy,
             qw * gy - qx * gz + qz * gx,
             qw * gz + qx * gy - qy * gx
        ], dtype=np.float64)

        # Normalize accelerometer measurement
        a_norm = np.linalg.norm(accel)
        if a_norm > 1e-4:
            ax, ay, az = accel / a_norm

            # Objective function gradient from gravity vector [0, 0, 1]
            # f(q, a) = 2*(qx*qz - qw*qy) - ax, 2*(qw*qx + qy*qz) - ay, 2*(0.5 - qx² - qy²) - az
            # ∇f = J^T * f
            f_1 = 2.0 * (qx * qz - qw * qy) - ax
            f_2 = 2.0 * (qw * qx + qy * qz) - ay
            f_3 = 2.0 * (0.5 - qx**2 - qy**2) - az

            # Jacobian J^T * f
            grad = np.array([
                -2.0 * qy * f_1 + 2.0 * qx * f_2,
                 2.0 * qz * f_1 + 2.0 * qw * f_2 - 4.0 * qx * f_3,
                -2.0 * qw * f_1 + 2.0 * qz * f_2 - 4.0 * qy * f_3,
                 2.0 * qx * f_1 + 2.0 * qy * f_2
            ], dtype=np.float64)

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-6:
                grad /= grad_norm

            # Combine gyro integration with accelerometer gradient correction
            q_dot = q_dot_omega - self.beta * grad
        else:
            q_dot = q_dot_omega

        # Integrate: q_new = q + q_dot * dt
        q = q + q_dot * self.dt
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            q /= q_norm

        self.q = q
        return self.q.copy()

    def process_sequence(self, gyro_seq: np.ndarray, accel_seq: np.ndarray) -> np.ndarray:
        """
        Processes an entire time sequence of shape (N, 3).
        Returns:
            Quaternions array of shape (N, 4)
        """
        n_samples = len(gyro_seq)
        quats = np.zeros((n_samples, 4), dtype=np.float32)
        for i in range(n_samples):
            quats[i] = self.update_imu(gyro_seq[i], accel_seq[i])
        return quats
