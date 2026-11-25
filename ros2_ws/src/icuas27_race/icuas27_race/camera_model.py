"""Pinhole model of the AI-deck camera and projection of the gates into its image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Gazebo cameras look along +x of their link; the optical frame has x right, y down, z forward.
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


@dataclass
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    t_body_cam: np.ndarray      # (3,) camera optical centre in the body frame
    r_body_link: np.ndarray     # (3, 3) camera link orientation in the body frame

    @staticmethod
    def from_camera_info(msg, t_body_cam, r_body_link=None) -> 'PinholeCamera':
        k = msg.k
        return PinholeCamera(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]),
                             width=int(msg.width), height=int(msg.height),
                             t_body_cam=np.asarray(t_body_cam, dtype=float),
                             r_body_link=np.eye(3) if r_body_link is None else np.asarray(r_body_link))

    def world_to_optical(self, points_world: np.ndarray, r_world_body: np.ndarray,
                         p_world_body: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> optical-frame points (N, 3)."""
        r_world_opt = r_world_body @ self.r_body_link @ R_LINK_OPTICAL
        p_world_cam = p_world_body + r_world_body @ self.t_body_cam
        return (np.asarray(points_world, dtype=float) - p_world_cam) @ r_world_opt

    def project(self, points_optical: np.ndarray, min_depth: float = 0.05
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Optical points (N, 3) -> pixels (N, 2) and a mask of points in front of the camera."""
        pts = np.asarray(points_optical, dtype=float)
        front = pts[:, 2] > min_depth
        z = np.where(front, pts[:, 2], 1.0)
        uv = np.stack([self.fx * pts[:, 0] / z + self.cx, self.fy * pts[:, 1] / z + self.cy], axis=1)
        return uv, front

    def polygon_visibility(self, uv: np.ndarray, front: np.ndarray) -> Tuple[float, float]:
        """(fraction of polygon vertices inside the image, polygon area in px^2) of a gate opening.

        Area is 0 unless every vertex is in front of the camera."""
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        frac = float(inside.mean()) if len(inside) else 0.0
        if not front.all():
            return frac, 0.0
        x, y = uv[:, 0], uv[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return frac, area


def hfov_focal(width: int, hfov: float) -> float:
    """Focal length in pixels of a Gazebo camera with horizontal field of view hfov (rad)."""
    return width / (2.0 * np.tan(0.5 * hfov))


def optional_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
