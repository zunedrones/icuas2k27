"""Rigid-body and distance helpers (numpy only, no ROS).

Conventions (see the project README):
  * ``T_a_b`` is the pose of frame ``b`` expressed in frame ``a``: ``p_a = T_a_b @ p_b``.
  * Quaternions are **xyzw** (ROS messages / scipy order) everywhere in this package.
  * Angles in radians, distances in metres. World frame is ENU, body frame is FLU.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot_z(yaw: float) -> np.ndarray:
    """Rotation matrix (3, 3) about +z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> rotation matrix (3, 3)."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_xyzw_to_matrices(q: np.ndarray) -> np.ndarray:
    """Batch version: (N, 4) xyzw -> (N, 3, 3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> unit quaternion (x, y, z, w) with w >= 0."""
    r = np.asarray(r, dtype=float)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 (xyzw)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Pure yaw rotation as xyzw."""
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) -> xyzw.

    Same convention as ignition::math::Quaterniond(roll, pitch, yaw) and tf2 setRPY.
    """
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def yaw_of_matrix(r: np.ndarray) -> float:
    """Heading of the body x axis projected on the horizontal plane (as the LUS managers do)."""
    return math.atan2(r[1, 0], r[0, 0])


def yaw_of_quat_xyzw(q: np.ndarray) -> float:
    return yaw_of_matrix(quat_xyzw_to_matrix(q))


def roll_pitch_of_matrix(r: np.ndarray) -> Tuple[float, float]:
    """Roll and pitch of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    return roll, pitch


def make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Homogeneous transform (4, 4) from rotation (3, 3) and translation (3,)."""
    tf = np.eye(4)
    tf[:3, :3] = r
    tf[:3, 3] = t
    return tf


def invert_transform(tf: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform without a general matrix inverse."""
    r = tf[:3, :3]
    t = tf[:3, 3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def transform_points(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply T (4, 4) to points (N, 3) or (3,)."""
    pts = np.asarray(points, dtype=float)
    return pts @ tf[:3, :3].T + tf[:3, 3]


def average_quaternions_xyzw(quats: np.ndarray) -> np.ndarray:
    """Mean rotation of nearby quaternions (N, 4): eigenvector method (Markley et al. 2007)."""
    q = np.asarray(quats, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    m = q.T @ q
    eigvals, eigvecs = np.linalg.eigh(m)
    mean = eigvecs[:, int(np.argmax(eigvals))]
    return mean if mean[3] >= 0.0 else -mean


def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                                c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point of each triangle (a[i], b[i], c[i]) to the point p.

    Vectorised form of Ericson, "Real-Time Collision Detection", sec. 5.1.5.
    p: (3,), a/b/c: (N, 3) -> (closest (N, 3), distance (N,)). Broadcasts: p (P, 1, 3) with
    a/b/c (N, 3) gives (P, N, 3) and (P, N).
    Degenerate (zero-area) triangles must be filtered out beforehand.
    """
    p = np.asarray(p, dtype=float)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum('...j,...j->...', ab, ap)
    d2 = np.einsum('...j,...j->...', ac, ap)
    bp = p - b
    d3 = np.einsum('...j,...j->...', ab, bp)
    d4 = np.einsum('...j,...j->...', ac, bp)
    cp = p - c
    d5 = np.einsum('...j,...j->...', ab, cp)
    d6 = np.einsum('...j,...j->...', ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    with np.errstate(divide='ignore', invalid='ignore'):
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        denom = 1.0 / (va + vb + vc)
    v_in = vb * denom
    w_in = vc * denom

    region_a = (d1 <= 0.0) & (d2 <= 0.0)
    region_b = (d3 >= 0.0) & (d4 <= d3)
    region_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    region_c = (d6 >= 0.0) & (d5 <= d6)
    region_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    region_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    e = (Ellipsis, None)
    closest = a + ab * v_in[e] + ac * w_in[e]
    # Apply regions in reverse priority so the first matching test of the scalar algorithm wins.
    closest = np.where(region_bc[e], b + (c - b) * w_bc[e], closest)
    closest = np.where(region_ac[e], a + ac * w_ac[e], closest)
    closest = np.where(region_c[e], c, closest)
    closest = np.where(region_ab[e], a + ab * v_ab[e], closest)
    closest = np.where(region_b[e], b, closest)
    closest = np.where(region_a[e], a, closest)
    dist = np.linalg.norm(closest - p, axis=-1)
    return closest, dist


def hull_support(normal_body_z: np.ndarray, direction: np.ndarray, radius: float,
                 half_height: float) -> float:
    """Extent of a flat cylinder (rotor disk hull) along a unit direction.

    The hull is a disk of `radius` in the body x-y plane, `2 * half_height` thick, centred
    on the body origin. normal_body_z is the body z axis in the world frame.
    """
    cos_t = float(abs(np.dot(normal_body_z, direction)))
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    return radius * sin_t + half_height * cos_t
