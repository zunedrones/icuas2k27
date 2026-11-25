"""Course, planner and referee checks with synthetic trajectories (no ROS, no simulator)."""

import math
import os

import numpy as np
import pytest

from icuas27_race.course import Course
from icuas27_race.geometry import make_transform, quat_xyzw_to_matrix, quat_from_rpy, rot_z
from icuas27_race.mesh import ObstacleField
from icuas27_race.planner import (FrameAlignment, PlanParams, ValidationParams, drop_close_waypoints,
                                  race_waypoints, recovery_waypoints, validate_trajectory)
from icuas27_race.referee_core import FINISHED, INVALID, RefereeCore, RefereeParams

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE_FILE = os.path.join(HERE, '..', '..', 'icuas27_bringup', 'config', 'courses',
                           'crazyflie_map.yaml')


@pytest.fixture(scope='module')
def course():
    c = Course.load(COURSE_FILE)
    c.obstacle_mesh = None   # keep the tests independent of the LUS installation
    return c


def path_samples(points, speed=2.0, rate=100.0):
    """Constant-speed polyline sampling: returns (t, p, v)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    ts, ps, vs = [], [], []
    t = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(b - a)
        if seg < 1e-9:
            continue
        direction = (b - a) / seg
        n = max(1, int(math.ceil(seg / speed * rate)))
        for k in range(n):
            ts.append(t + k / rate)
            ps.append(a + direction * seg * k / n)
            vs.append(direction * speed)
        t += n / rate
    ts.append(t)
    ps.append(pts[-1])
    vs.append(np.zeros(3))
    return np.asarray(ts), np.asarray(ps), np.asarray(vs)


def run_referee(core, t, p, v, t0=100.0, rot=None):
    rot = np.eye(3) if rot is None else rot
    events = []
    for k in range(len(t)):
        events += core.update(t0 + t[k], p[k], v[k], rot)
    return events


def test_course_geometry(course):
    assert course.laps == 7
    assert len(course.sequence) == 49
    assert course.sequence[:8] == [1, 2, 3, 4, 5, 6, 7, 1]
    for gate in course.gates.values():
        assert np.allclose(gate.to_gate(gate.center + 0.1 * gate.normal), [0.1, 0.0, 0.0])
        assert np.allclose(gate.to_world(gate.to_gate(np.array([1.0, 2.0, 3.0]))), [1.0, 2.0, 3.0])
        assert course.inside_arena(gate.center)
        assert gate.inner_width == pytest.approx(0.5) and gate.outer_width == pytest.approx(0.6)
    # travel direction of G1 is -y (away from the takeoff base)
    assert np.allclose(course.gates[1].normal, [0.0, -1.0, 0.0], atol=1e-6)
    assert course.over_pad(np.array([0.1, -0.1, 0.1]))
    assert not course.over_pad(np.array([0.4, 0.0, 0.1]))


def test_race_waypoints(course):
    params = PlanParams()
    wps = race_waypoints(course, params)
    assert len(wps) == 49 * 3 + 1 <= 256          # planning-manager mission limit
    assert wps[-1].kind == 'final'
    g1 = course.gates[1]
    assert np.allclose(wps[0].position, g1.center - 0.3 * g1.normal)
    assert np.allclose(wps[1].position, g1.center)
    for a, b in zip(wps[:-1], wps[1:]):
        assert np.linalg.norm(a.position - b.position) > 0.05
    gate_wps = [w for w in wps if w.kind == 'gate']
    assert [w.sequence_index for w in gate_wps] == list(range(49))
    assert drop_close_waypoints(wps, wps[0].position)[0].label == wps[1].label


def test_recovery_goes_around_the_gate(course):
    params = PlanParams()
    k = 9                                   # lap 2, gate 3
    gate = course.gates[course.sequence[k]]
    past = gate.to_world(np.array([0.5, 0.4, 0.0]))     # beyond the plane, left side
    wps = recovery_waypoints(course, params, k, past)
    assert wps[0].kind == 'recovery' and wps[1].kind == 'recovery'
    local = [gate.to_gate(w.position) for w in wps[:3]]
    assert local[0][0] > 0 and local[1][0] < 0          # exit side, then entry side
    assert abs(local[0][1]) == pytest.approx(params.recovery_lateral)
    assert np.sign(local[0][1]) == 1                    # same side as the drone
    assert wps[2].label.endswith('-pre') and wps[2].sequence_index == k
    before = gate.to_world(np.array([-0.8, 0.0, 0.0]))
    wps2 = recovery_waypoints(course, params, k, before)
    assert wps2[0].kind == 'pre'


def test_frame_alignment_recovers_transform():
    t_wo = make_transform(rot_z(0.3), np.array([1.0, -2.0, 0.07]))
    rng = np.random.default_rng(1)
    world_body, odom_body = [], []
    for _ in range(20):
        t_ob = make_transform(quat_xyzw_to_matrix(quat_from_rpy(*rng.normal(scale=0.1, size=3))),
                              rng.normal(size=3))
        odom_body.append(t_ob)
        world_body.append(t_wo @ t_ob)
    al = FrameAlignment.from_pose_pairs(world_body, odom_body)
    assert np.allclose(al.t_world_odom, t_wo, atol=1e-9)
    p = np.array([0.5, 0.2, 1.0])
    assert np.allclose(al.point_to_world(al.point_to_odom(p)), p)
    assert al.heading_to_world(al.heading_to_odom(1.0)) == pytest.approx(1.0)


def test_validation_flags_walls(course):
    t, p, v = path_samples([[0.0, 0.0, 1.0], [-2.0, -2.0, 1.0]])
    ok = validate_trajectory(course, None, p, v, t, ValidationParams())
    assert ok.ok, ok.violations
    t, p, v = path_samples([[0.0, 0.0, 1.0], [0.45, 0.0, 1.0]])
    bad = validate_trajectory(course, None, p, v, t, ValidationParams())
    assert not bad.ok and 'wall' in bad.violations[0]


def full_race_points(course, params):
    pts = [np.array([0.0, 0.0, 0.10]), np.array([0.0, 0.0, 1.0])]
    pts += [w.position for w in race_waypoints(course, params)]
    pts += [np.array([0.0, 0.0, 0.10])]
    return pts


def test_referee_scores_a_clean_race(course):
    core = RefereeCore(course, RefereeParams(time_limit=600.0))
    rest = np.array([0.0, 0.0, 0.10])
    for k in range(20):                                  # resting on the base before the start
        core.update(99.0 + 0.05 * k, rest, np.zeros(3), np.eye(3))
    core.start(100.0)
    t, p, v = path_samples(full_race_points(course, PlanParams()), speed=2.0)
    t = np.append(t, t[-1] + np.arange(1, 60) * 0.01)    # stay landed
    p = np.vstack([p, np.repeat(p[-1:], 59, axis=0)])
    v = np.vstack([v, np.zeros((59, 3))])
    events = run_referee(core, t, p, v)
    s = core.summary()
    assert core.state == FINISHED, (s['reason'], [e.type for e in events][-5:])
    assert s['valid'] and s['gates_passed'] == 49 and s['laps_completed'] == 7
    assert s['hits'] == 0 and s['missed_entries'] == 0 and s['wrong_way'] == 0
    assert s['elapsed_time'] == pytest.approx(t[-60], abs=0.02)
    assert s['adjusted_time'] == pytest.approx(s['elapsed_time'])
    assert len(s['lap_times']['by_last_gate']) == 7
    assert len(s['lap_times']['flying']) == 6
    for ps in s['passes']:
        assert abs(ps['offset_y']) < 1e-6 and abs(ps['offset_z']) < 1e-6
        assert ps['approach_angle'] < 1e-3          # straight through along the normal


def test_referee_detects_missed_and_wrong_way(course):
    g1 = course.gates[1]
    core = RefereeCore(course, RefereeParams())
    core.update(99.0, np.array([0.0, 0.0, 0.1]), np.zeros(3), np.eye(3))
    core.start(100.0)
    beside = [g1.to_world(np.array([-0.5, 0.45, 0.0])), g1.to_world(np.array([0.5, 0.45, 0.0]))]
    t, p, v = path_samples(beside)
    events = run_referee(core, t, p, v)
    assert [e.type for e in events] == ['MISSED']
    assert core.missed_entries == 1 and core.next_index == 0
    back = [g1.to_world(np.array([0.5, 0.0, 0.0])), g1.to_world(np.array([-0.5, 0.0, 0.0]))]
    t, p, v = path_samples(back)
    events = run_referee(core, t, p, v, t0=110.0)
    assert [e.type for e in events] == ['WRONG_WAY']
    through = [g1.to_world(np.array([-0.5, 0.0, 0.0])), g1.to_world(np.array([0.5, 0.0, 0.0]))]
    t, p, v = path_samples(through)
    events = run_referee(core, t, p, v, t0=120.0)
    assert [e.type for e in events] == ['PASS'] and events[0].detail.startswith('retry')
    assert core.next_index == 1
    assert core.penalty_time == pytest.approx(5.0)


def test_referee_out_of_bounds_invalidates(course):
    core = RefereeCore(course, RefereeParams())
    core.update(99.0, np.array([0.0, 0.0, 0.1]), np.zeros(3), np.eye(3))
    core.start(100.0)
    t, p, v = path_samples([[0.0, 0.0, 1.0], [0.8, 0.0, 1.0]])
    events = run_referee(core, t, p, v)
    assert core.state == INVALID and events[-1].type == 'OUT_OF_BOUNDS'


def test_referee_detects_contact_with_frame(course):
    g1 = course.gates[1]
    # top bar of gate 1 as two triangles (frame plane, z in [0.25, 0.30] in the gate frame)
    quad = np.array([[0.0, -0.3, 0.25], [0.0, 0.3, 0.25], [0.0, 0.3, 0.30], [0.0, -0.3, 0.30]])
    w = g1.to_world(quad)
    tris = np.array([[w[0], w[1], w[2]], [w[0], w[2], w[3]]])
    field = ObstacleField.from_triangles(tris)
    core = RefereeCore(course, RefereeParams(), field)
    core.update(99.0, np.array([0.0, 0.0, 0.1]), np.zeros(3), np.eye(3))
    core.start(100.0)
    # crossing 1 cm below the bar: the 2.5 cm hull half-height touches it
    t, p, v = path_samples([g1.to_world(np.array([-0.5, 0.0, 0.24])),
                            g1.to_world(np.array([0.5, 0.0, 0.24]))])
    events = run_referee(core, t, p, v)
    kinds = [e.type for e in events]
    assert 'HIT' in kinds and 'MISSED' in kinds and 'PASS' not in kinds
    assert core.hits == 1 and core.missed_entries == 1
    assert core.penalty_time == pytest.approx(10.0)


def test_referee_crash_on_floor(course):
    core = RefereeCore(course, RefereeParams())
    core.update(99.0, np.array([0.0, 0.0, 0.1]), np.zeros(3), np.eye(3))
    core.start(100.0)
    t, p, v = path_samples([[0.0, 0.0, 0.1], [0.0, 0.0, 1.0], [-1.0, -2.0, 1.0],
                            [-1.0, -2.0, 0.05]])
    run_referee(core, t, p, v)
    assert core.state == INVALID and 'floor' in core.reason


def test_receding_horizon_slices(course):
    params = PlanParams()
    head = race_waypoints(course, params, 0, 9)
    assert len(head) == 9 * 3 and head[-1].kind == 'post'
    tail = race_waypoints(course, params, 44, 53)
    assert len(tail) == 5 * 3 + 1 and tail[-1].kind == 'final'
    assert [w.label for w in race_waypoints(course, params)] == \
        [w.label for w in race_waypoints(course, params, 0, 9) + race_waypoints(course, params, 9)]


def test_trim_passed_waypoints(course):
    from icuas27_race.planner import trim_passed_waypoints
    params = PlanParams()
    wps = race_waypoints(course, params, 6, 9)          # G7 (lap 1), then G1, G2 of lap 2
    g7 = course.gates[7]
    assert len(trim_passed_waypoints(course, wps, g7.to_world(np.array([-1.0, 0, 0])))) == len(wps)
    between = trim_passed_waypoints(course, wps, g7.to_world(np.array([-0.2, 0, 0])))
    assert between[0].kind == 'gate' and between[0].sequence_index == 6
    past = trim_passed_waypoints(course, wps, g7.to_world(np.array([0.1, 0, 0])))
    assert past[0].kind == 'post'
    far = trim_passed_waypoints(course, wps, g7.to_world(np.array([0.6, 0, 0])))
    assert far[0].sequence_index == 7 and far[0].kind == 'pre'
