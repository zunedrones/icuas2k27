"""ROS 2 helpers shared by the race nodes: QoS profiles, message conversions and the Gazebo
ground-truth adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)

from .geometry import (quat_conjugate_xyzw, quat_from_rpy, quat_multiply_xyzw,
                       quat_xyzw_to_matrix, make_transform)

# Streams: best effort is compatible with both reliable and best-effort publishers.
SENSOR_QOS = qos_profile_sensor_data
STATUS_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
# Latched state (LUS planning diagnostics, our race events and plan).
LATCHED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
EVENTS_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        history=HistoryPolicy.KEEP_LAST)
RELIABLE_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def point_to_np(p) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def vec_to_np(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)


def quat_to_np(q) -> np.ndarray:
    """geometry_msgs/Quaternion -> xyzw array."""
    return np.array([q.x, q.y, q.z, q.w], dtype=float)


def pose_to_transform(pose) -> np.ndarray:
    return make_transform(quat_xyzw_to_matrix(quat_to_np(pose.orientation)),
                          point_to_np(pose.position))


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


@dataclass
class WorldState:
    stamp: float
    position: np.ndarray         # world (3,)
    quat: np.ndarray             # world xyzw (4,)
    rotation: np.ndarray         # R_world_body (3, 3)
    velocity: np.ndarray         # world (3,)
    angular_velocity: np.ndarray  # body (3,)

    @property
    def transform(self) -> np.ndarray:
        return make_transform(self.rotation, self.position)


class GroundTruthAdapter:
    """Undo the constant offsets of the LASER P3D ground-truth plugin.

    laser_uav_simulation/scripts/jinja_gen.py sets the plugin offsets to minus the spawn pose
    (xyz_offset = -[x, y, z], rpy_offset = -[roll, pitch, yaw]) and the plugin reports
    p_gt = p_world + xyz_offset and q_gt = q(rpy_offset) * q_world. Its twist is expressed in
    the body frame (local_twist = true). This class returns the true world-frame state.
    """

    def __init__(self, spawn_position, spawn_rpy, twist_in_body: bool = True):
        self.spawn_position = np.asarray(spawn_position, dtype=float)
        roll, pitch, yaw = (float(x) for x in spawn_rpy)
        self.q_offset_inv = quat_conjugate_xyzw(quat_from_rpy(-roll, -pitch, -yaw))
        self.twist_in_body = twist_in_body

    def convert(self, msg) -> WorldState:
        pose = msg.pose.pose
        p_world = point_to_np(pose.position) + self.spawn_position
        q_world = quat_multiply_xyzw(self.q_offset_inv, quat_to_np(pose.orientation))
        q_world /= np.linalg.norm(q_world)
        r = quat_xyzw_to_matrix(q_world)
        v = vec_to_np(msg.twist.twist.linear)
        w = vec_to_np(msg.twist.twist.angular)
        v_world = r @ v if self.twist_in_body else v
        w_body = w if self.twist_in_body else r.T @ w
        return WorldState(stamp_to_sec(msg.header.stamp), p_world, q_world, r, v_world, w_body)


def odometry_state(msg) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """LUS estimation odometry -> (stamp, position, R, velocity in the odometry frame).

    The LUS hardware API and estimation manager publish the linear twist in the body (FLU)
    frame, so it is rotated into the odometry frame here.
    """
    pose = msg.pose.pose
    r = quat_xyzw_to_matrix(quat_to_np(pose.orientation))
    v_body = vec_to_np(msg.twist.twist.linear)
    return stamp_to_sec(msg.header.stamp), point_to_np(pose.position), r, r @ v_body
