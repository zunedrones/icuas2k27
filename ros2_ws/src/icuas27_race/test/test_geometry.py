"""Numerical checks of the rigid-body and distance helpers."""

import math

import numpy as np
import pytest

from icuas27_race.geometry import (average_quaternions_xyzw, closest_points_on_triangles,
                                   hull_support, invert_transform, make_transform,
                                   matrix_to_quat_xyzw, quat_conjugate_xyzw, quat_from_rpy,
                                   quat_from_yaw, quat_multiply_xyzw, quat_xyzw_to_matrices,
                                   quat_xyzw_to_matrix, roll_pitch_of_matrix, rot_z,
                                   transform_points, wrap_angle, yaw_of_matrix)

RNG = np.random.default_rng(27)


def random_quat():
    q = RNG.normal(size=4)
    return q / np.linalg.norm(q)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_quaternion_matrix_round_trip():
    for _ in range(200):
        q = random_quat()
        r = quat_xyzw_to_matrix(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)
        q2 = matrix_to_quat_xyzw(r)
        assert np.isclose(abs(np.dot(q, q2)), 1.0, atol=1e-12)
        assert np.allclose(quat_xyzw_to_matrix(q2), r, atol=1e-12)


def test_batch_matrices_match_single():
    qs = np.stack([random_quat() for _ in range(20)])
    batch = quat_xyzw_to_matrices(qs)
    for q, r in zip(qs, batch):
        assert np.allclose(r, quat_xyzw_to_matrix(q), atol=1e-12)


def test_rpy_convention_is_fixed_axis_zyx():
    for _ in range(50):
        roll, pitch, yaw = RNG.uniform(-1.4, 1.4, size=3)
        expected = rot_z(yaw) @ ry(pitch) @ rx(roll)
        assert np.allclose(quat_xyzw_to_matrix(quat_from_rpy(roll, pitch, yaw)), expected,
                           atol=1e-12)
        r2, p2 = roll_pitch_of_matrix(expected)
        assert np.isclose(r2, roll) and np.isclose(p2, pitch)
        assert np.isclose(yaw_of_matrix(expected), wrap_angle(yaw))


def test_quaternion_product_matches_matrix_product():
    for _ in range(50):
        a, b = random_quat(), random_quat()
        ab = quat_multiply_xyzw(a, b)
        assert np.allclose(quat_xyzw_to_matrix(ab), quat_xyzw_to_matrix(a) @ quat_xyzw_to_matrix(b),
                           atol=1e-12)
        ident = quat_multiply_xyzw(a, quat_conjugate_xyzw(a))
        assert np.allclose(ident, [0, 0, 0, 1], atol=1e-12)


def test_yaw_quaternion():
    for yaw in np.linspace(-3.0, 3.0, 13):
        assert np.allclose(quat_xyzw_to_matrix(quat_from_yaw(yaw)), rot_z(yaw), atol=1e-12)


def test_transform_inverse_and_points():
    for _ in range(50):
        t = make_transform(quat_xyzw_to_matrix(random_quat()), RNG.normal(size=3))
        assert np.allclose(t @ invert_transform(t), np.eye(4), atol=1e-12)
        pts = RNG.normal(size=(10, 3))
        back = transform_points(invert_transform(t), transform_points(t, pts))
        assert np.allclose(back, pts, atol=1e-12)


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * math.pi), -math.pi)
    assert np.isclose(wrap_angle(-math.pi / 2), -math.pi / 2)
    assert -math.pi <= wrap_angle(7.0) < math.pi


def test_average_quaternions_of_noisy_samples():
    q0 = random_quat()
    samples = []
    for _ in range(100):
        noise = quat_from_rpy(*RNG.normal(scale=0.01, size=3))
        q = quat_multiply_xyzw(q0, noise)
        samples.append(q if RNG.random() > 0.5 else -q)  # sign ambiguity must not matter
    mean = average_quaternions_xyzw(np.asarray(samples))
    assert abs(np.dot(mean, q0)) > 1 - 1e-4


def brute_force_distance(p, a, b, c, n=120):
    best = math.inf
    for i in range(n + 1):
        for j in range(n + 1 - i):
            u, v = i / n, j / n
            q = a + u * (b - a) + v * (c - a)
            best = min(best, float(np.linalg.norm(p - q)))
    return best


@pytest.mark.parametrize('seed', range(8))
def test_closest_point_on_triangle_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    tris = rng.normal(size=(6, 3, 3))
    for _ in range(5):
        p = rng.normal(size=3) * 1.5
        cp, dist = closest_points_on_triangles(p, tris[:, 0], tris[:, 1], tris[:, 2])
        for k in range(len(tris)):
            ref = brute_force_distance(p, *tris[k])
            assert dist[k] <= ref + 1e-9           # never worse than any sampled point
            assert dist[k] >= ref - 0.03           # and within the sampling resolution
            assert np.isclose(np.linalg.norm(cp[k] - p), dist[k])


def test_hull_support():
    up = np.array([0.0, 0.0, 1.0])
    assert np.isclose(hull_support(up, np.array([1.0, 0.0, 0.0]), 0.07, 0.025), 0.07)
    assert np.isclose(hull_support(up, np.array([0.0, 0.0, -1.0]), 0.07, 0.025), 0.025)
    d = np.array([1.0, 0.0, 1.0]) / math.sqrt(2)
    assert np.isclose(hull_support(up, d, 0.07, 0.025), (0.07 + 0.025) / math.sqrt(2))


def test_batched_closest_points_match_single():
    rng = np.random.default_rng(3)
    tris = rng.normal(size=(9, 3, 3))
    pts = rng.normal(size=(7, 3)) * 2
    cp_b, d_b = closest_points_on_triangles(pts[:, None, :], tris[:, 0], tris[:, 1], tris[:, 2])
    assert d_b.shape == (7, 9)
    for i, p in enumerate(pts):
        cp, d = closest_points_on_triangles(p, tris[:, 0], tris[:, 1], tris[:, 2])
        assert np.allclose(d_b[i], d) and np.allclose(cp_b[i], cp)


def test_min_distance_batch_matches_brute_force():
    from icuas27_race.mesh import ObstacleField
    rng = np.random.default_rng(11)
    # two thin "frames" of random triangles around different centres
    a = rng.normal(scale=[0.05, 0.4, 0.4], size=(40, 3, 3)) + np.array([0.0, 0.0, 1.0])
    b = rng.normal(scale=[0.4, 0.05, 0.4], size=(40, 3, 3)) + np.array([2.0, 1.0, 1.0])
    field = ObstacleField.from_triangles(np.concatenate([a, b]),
                                         {'a': np.array([0.0, 0.0, 1.0]),
                                          'b': np.array([2.0, 1.0, 1.0])})
    tris = np.concatenate([c.triangles for c in field.clusters])
    pts = rng.uniform([-1, -1, 0], [3, 2, 2], size=(300, 3))
    max_range = 0.5
    got = field.min_distance_batch(pts, max_range=max_range)
    _, d = closest_points_on_triangles(pts[:, None, :], tris[:, 0], tris[:, 1], tris[:, 2])
    exact = d.min(axis=1)
    near = exact <= max_range
    assert np.allclose(got[near], exact[near])          # exact wherever it matters
    assert np.all(got[~near] >= exact[~near] - 1e-12)   # never underestimates otherwise
