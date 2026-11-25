"""Checks of the AI-deck pinhole model (optical frame: x right, y down, z forward)."""

import math

import numpy as np
import pytest

from icuas27_race.camera_model import R_LINK_OPTICAL, PinholeCamera, hfov_focal
from icuas27_race.geometry import rot_z


def camera():
    f = hfov_focal(324, 1.5184)          # AI-deck HM01B0, 87 deg horizontal FOV
    return PinholeCamera(fx=f, fy=f, cx=162.0, cy=122.0, width=324, height=244,
                         t_body_cam=np.array([0.029, 0.0, 0.015]), r_body_link=np.eye(3))


def test_link_to_optical_rotation_is_proper():
    assert np.allclose(R_LINK_OPTICAL @ R_LINK_OPTICAL.T, np.eye(3))
    assert np.isclose(np.linalg.det(R_LINK_OPTICAL), 1.0)
    # optical z (forward) is the link x axis, optical x (right) is -link y, optical y (down) -link z
    assert np.allclose(R_LINK_OPTICAL[:, 2], [1, 0, 0])
    assert np.allclose(R_LINK_OPTICAL[:, 0], [0, -1, 0])
    assert np.allclose(R_LINK_OPTICAL[:, 1], [0, 0, -1])


def test_projection_directions():
    cam = camera()
    yaw = 0.7
    r = rot_z(yaw)
    p = np.array([1.0, 2.0, 1.0])
    lens = p + r @ cam.t_body_cam
    fwd, left, up = r[:, 0], r[:, 1], r[:, 2]
    pts = np.stack([lens + 2.0 * fwd, lens + 2.0 * fwd + 0.3 * left, lens + 2.0 * fwd + 0.3 * up,
                    lens - 1.0 * fwd])
    uv, front = cam.project(cam.world_to_optical(pts, r, p))
    assert front.tolist() == [True, True, True, False]
    assert np.allclose(uv[0], [cam.cx, cam.cy])
    assert uv[1][0] < cam.cx and np.isclose(uv[1][1], cam.cy)       # left -> smaller u
    assert uv[2][1] < cam.cy and np.isclose(uv[2][0], cam.cx)       # up -> smaller v
    assert uv[1][0] == pytest.approx(cam.cx - cam.fx * 0.3 / 2.0)


def test_gate_visibility():
    cam = camera()
    r = np.eye(3)
    p = np.zeros(3)
    lens = p + cam.t_body_cam
    d = 1.5
    square = np.array([[d, -0.25, -0.25], [d, 0.25, -0.25], [d, 0.25, 0.25], [d, -0.25, 0.25]]) + lens
    uv, front = cam.project(cam.world_to_optical(square, r, p))
    frac, area = cam.polygon_visibility(uv, front)
    assert frac == 1.0
    assert area == pytest.approx((cam.fx * 0.5 / d) ** 2, rel=1e-9)
    assert math.isclose(hfov_focal(324, 1.5184), 324 / (2 * math.tan(0.7592)))
