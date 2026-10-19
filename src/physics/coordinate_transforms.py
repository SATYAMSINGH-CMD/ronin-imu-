"""
Physics Pre-Processing & Coordinate Transformations
1. Quaternions -> 3x3 Rotation Matrix
2. Body Frame -> World Frame Rotation
3. Clean Gravity Subtraction (9.80665 m/s²)
4. Heading-Independent Frame (HIF) Alignment
5. Stationary / Zero Velocity (ZUPT) Detection
"""

import numpy as np


def quats_to_rot_matrices(quats: np.ndarray) -> np.ndarray:
    """
    Converts quaternions [qw, qx, qy, qz] into 3x3 rotation matrices.
    Supports single quaternion (4,) or batch/sequence (N, 4).
    Returns:
        Rotation matrix of shape (3, 3) or (N, 3, 3).
        R rotates vectors from Body Frame to World Frame: v_world = R * v_body
    """
    single = False
    if quats.ndim == 1:
        quats = quats[None, :]
        single = True

    qw = quats[:, 0]
    qx = quats[:, 1]
    qy = quats[:, 2]
    qz = quats[:, 3]

    # Pre-compute products
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz

    N = len(quats)
    R = np.zeros((N, 3, 3), dtype=np.float32)

    R[:, 0, 0] = 1.0 - 2.0 * (qy2 + qz2)
    R[:, 0, 1] = 2.0 * (qxqy - qwqz)
    R[:, 0, 2] = 2.0 * (qxqz + qwqy)

    R[:, 1, 0] = 2.0 * (qxqy + qwqz)
    R[:, 1, 1] = 1.0 - 2.0 * (qx2 + qz2)
    R[:, 1, 2] = 2.0 * (qyqz - qwqx)

    R[:, 2, 0] = 2.0 * (qxqz - qwqy)
    R[:, 2, 1] = 2.0 * (qyqz + qwqx)
    R[:, 2, 2] = 1.0 - 2.0 * (qx2 + qy2)

    return R[0] if single else R


def rotate_body_to_world(vectors_body: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """
    Rotates 3D vectors (e.g. acceleration or gyro) from sensor body frame to world frame.
    v_world = R(q) * v_body
    Args:
        vectors_body: Shape (N, 3)
        quats: Shape (N, 4)
    Returns:
        vectors_world: Shape (N, 3)
    """
    R = quats_to_rot_matrices(quats)  # (N, 3, 3)
    # Batch matrix multiplication: (N, 3, 3) @ (N, 3, 1) -> (N, 3)
    vectors_world = np.einsum("nij,nj->ni", R, vectors_body)
    return vectors_world


def subtract_gravity(accel_world: np.ndarray, g: float = 9.80665) -> np.ndarray:
    """
    Clean gravity subtraction from World Frame acceleration.
    Since Z is pointing up toward sky, gravity acts downward:
        a_linear = a_world - [0, 0, +9.80665]
    Returns:
        accel_linear: Shape (N, 3)
    """
    accel_linear = accel_world.copy()
    accel_linear[:, 2] -= g
    return accel_linear


def extract_yaw_from_quats(quats: np.ndarray) -> np.ndarray:
    """
    Extracts Compass Yaw angle (around Z axis) in radians from quaternions.
    Yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
    """
    qw = quats[:, 0]
    qx = quats[:, 1]
    qy = quats[:, 2]
    qz = quats[:, 3]

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def project_to_heading_independent_frame(
    accel_world_linear: np.ndarray,
    gyro_world: np.ndarray,
    anchor_yaw: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heading-Independent Frame (HIF):
    Rotates horizontal X and Y by -anchor_yaw so that window starts facing 0°.
    Vertical Z is unaffected.
    """
    cos_psi = np.cos(-anchor_yaw)
    sin_psi = np.sin(-anchor_yaw)

    # 2D horizontal rotation matrix
    R_hif = np.array([
        [cos_psi, -sin_psi],
        [sin_psi,  cos_psi]
    ], dtype=np.float32)

    a_hif = accel_world_linear.copy()
    w_hif = gyro_world.copy()

    # Rotate horizontal components
    a_hif[:, :2] = accel_world_linear[:, :2] @ R_hif.T
    w_hif[:, :2] = gyro_world[:, :2] @ R_hif.T

    return a_hif, w_hif


def detect_stationary_periods(
    accel: np.ndarray,
    gyro: np.ndarray,
    window_samples: int = 40,
    var_threshold: float = 0.08
) -> np.ndarray:
    """
    Detects stationary periods (ZUPT) using moving variance of acceleration and gyroscope.
    Vectorized computation using scipy.ndimage.uniform_filter1d for high speed on large sequences.
    """
    from scipy.ndimage import uniform_filter1d

    # Var(X) = E[X^2] - (E[X])^2
    a_mean = uniform_filter1d(accel, size=window_samples, axis=0, mode="nearest")
    a_sq_mean = uniform_filter1d(accel ** 2, size=window_samples, axis=0, mode="nearest")
    a_var = np.maximum(0.0, a_sq_mean - a_mean ** 2).sum(axis=1)

    g_mean = uniform_filter1d(gyro, size=window_samples, axis=0, mode="nearest")
    g_sq_mean = uniform_filter1d(gyro ** 2, size=window_samples, axis=0, mode="nearest")
    g_var = np.maximum(0.0, g_sq_mean - g_mean ** 2).sum(axis=1)

    energy = a_var + 10.0 * g_var
    return energy < var_threshold
