"""End-to-end check of the analytics pipeline on a synthetic run directory."""

import csv
import json
import os

import numpy as np
import pytest

pytest.importorskip('matplotlib')

from icuas27_race.analytics import compare, report  # noqa: E402
from icuas27_race.course import Course  # noqa: E402
from icuas27_race.planner import PlanParams  # noqa: E402
from icuas27_race.referee_core import RefereeCore, RefereeParams  # noqa: E402
from test_race_core import COURSE_FILE, full_race_points, path_samples  # noqa: E402


def make_run(tmp_path, name):
    course = Course.load(COURSE_FILE)
    course.obstacle_mesh = None
    core = RefereeCore(course, RefereeParams(time_limit=600.0))
    core.update(99.0, np.array([0.0, 0.0, 0.1]), np.zeros(3), np.eye(3))
    core.start(100.0)
    t, p, v = path_samples(full_race_points(course, PlanParams()), speed=2.2)
    t = np.append(t, t[-1] + np.arange(1, 60) * 0.01)
    p = np.vstack([p, np.repeat(p[-1:], 59, axis=0)])
    v = np.vstack([v, np.zeros((59, 3))])
    for k in range(len(t)):
        core.update(100.0 + t[k], p[k], v[k], np.eye(3))
    run = tmp_path / name
    (run / 'referee').mkdir(parents=True)
    (run / 'plan').mkdir()
    summary = core.summary()
    summary['course'] = course.to_dict()
    summary['referee_params'] = vars(core.p)
    (run / 'referee' / 'summary.json').write_text(json.dumps(summary, default=float))
    with open(run / 'referee' / 'trajectory.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['stamp', 'race_time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw',
                    'wx', 'wy', 'wz', 'speed', 'clearance'])
        for k in range(len(t)):
            w.writerow([100.0 + t[k], t[k], *p[k], *v[k], 0, 0, 0, 1, 0, 0, 0,
                        np.linalg.norm(v[k]), 0.2])
    plan = {'params': {'race_speed': 3.0}, 'waypoints': [], 'alignment': {
        'source': 'identity', 'T_world_odom': np.eye(4).tolist(), 'yaw': 0.0},
        'validation': {'ok': True, 'min_clearance': 0.15, 'min_wall_margin': 0.3,
                       'violations': []},
        'prediction': {'duration': float(t[-60]), 'planning_ms': 5.0, 't': t[::5].tolist(),
                       'p': p[::5].tolist(), 'v': v[::5].tolist(), 'a': np.zeros_like(p[::5]).tolist(),
                       'heading': [0.0] * len(t[::5])}}
    (run / 'plan' / 'plan_race.json').write_text(json.dumps(plan))
    (run / 'telemetry').mkdir()
    (run / 'telemetry' / 'race.csv').write_text(
        'rx,source,state,elapsed,detail\n'
        '99.0,race_manager,READY,0.0,\n'
        '100.0,race_manager,ARMING,0.0,\n'
        '100.5,race_manager,TAKING_OFF,0.5,\n'
        '101.0,referee,1,1.0,\n'
        '103.0,race_manager,RACING,3.0,\n'
        f'{100.0 + t[-80]:.3f},race_manager,LANDING,0.0,\n'
        f'{100.0 + t[-1]:.3f},race_manager,FINISHED,0.0,\n')
    (run / 'plan' / 'missions.csv').write_text('tag,stamp,waypoints,predicted_duration\n'
                                               f'race,100.0,148,{t[-60]:.3f}\n')
    return run


def test_report_and_compare(tmp_path):
    runs = [make_run(tmp_path, 'run_a'), make_run(tmp_path, 'run_b')]
    s = json.load(open(runs[0] / 'referee' / 'summary.json'))
    traj = np.loadtxt(runs[0] / 'referee' / 'trajectory.csv', delimiter=',', skiprows=1)
    t_landing = s['elapsed_time'] - (traj[-80, 0] - 100.0)
    path = report.make_report(str(runs[0]))
    html = open(path, encoding='utf-8').read()
    assert 'VALID' in html and 'REPLAY_DATA' in html and 'Lap times' not in html[:100]
    metrics = json.load(open(os.path.join(runs[0], 'report', 'metrics.json')))
    assert metrics['valid'] and metrics['gates_passed'] == 49
    assert metrics['best_lap'] > 0 and metrics['max_speed'] == pytest.approx(2.2, abs=1e-6)
    assert metrics['phase_arming_s'] == pytest.approx(0.5)
    assert metrics['phase_taking_off_s'] == pytest.approx(2.5)
    assert metrics['phase_landing_s'] == pytest.approx(t_landing, abs=0.02)
    assert 'Time breakdown' in html
    assert os.path.exists(os.path.join(runs[0], 'report', 'figures', 'topdown.png'))
    assert compare.main([str(tmp_path)]) == 0
    assert os.path.exists(tmp_path / 'compare.html')
