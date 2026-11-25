"""Derived race metrics (laps, gates, dynamics, tracking, actuation, timing) from a RunData."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..geometry import roll_pitch_of_matrix, transform_points, yaw_of_matrix
from .rundata import RunData


def finite(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def smooth_derivative(t: np.ndarray, x: np.ndarray, half_window: int = 2) -> np.ndarray:
    """Central difference over +/- half_window samples (robust to 100 Hz jitter)."""
    n = len(t)
    out = np.zeros_like(x)
    for i in range(n):
        a, b = max(0, i - half_window), min(n - 1, i + half_window)
        dt = t[b] - t[a]
        out[i] = (x[b] - x[a]) / dt if dt > 1e-9 else 0.0
    return out


def attitude_deg(run: RunData) -> np.ndarray:
    """(N, 3) roll, pitch, yaw in degrees from the ground-truth quaternions."""
    rs = run.rotations
    out = np.empty((len(rs), 3))
    for i, r in enumerate(rs):
        roll, pitch = roll_pitch_of_matrix(r)
        out[i] = (math.degrees(roll), math.degrees(pitch), math.degrees(yaw_of_matrix(r)))
    return out


def race_window(run: RunData) -> np.ndarray:
    """Mask of trajectory samples between the start command and the end of the run."""
    t = run.t
    end = run.summary.get('elapsed_time') or (t[-1] if len(t) else 0.0)
    return (t >= 0.0) & (t <= end + 1e-6)


def lap_table(run: RunData) -> List[dict]:
    s = run.summary
    passes = s.get('passes', [])
    gpl = run.course.gates_per_lap
    t, p, v = run.t, run.p, run.v
    speed = np.linalg.norm(v, axis=1)
    clear = run.traj.get('clearance', np.full(len(t), np.nan))
    by_index = {ps['index']: ps for ps in passes}
    rows = []
    t_prev = 0.0
    for lap in range(run.course.laps):
        last = (lap + 1) * gpl - 1
        if last not in by_index:
            break
        t_end = float(by_index[last]['race_time'])
        m = (t >= t_prev) & (t <= t_end)
        lap_passes = [ps for ps in passes if lap * gpl <= ps['index'] <= last]
        offs = [math.hypot(ps['offset_y'], ps['offset_z']) for ps in lap_passes]
        seg = p[m]
        rows.append({
            'lap': lap + 1,
            'lap_time': t_end - t_prev,
            'flying_lap_time': (float(by_index[(lap + 1) * gpl]['race_time']) -
                                float(by_index[lap * gpl]['race_time']))
            if (lap + 1) * gpl in by_index and lap * gpl in by_index else None,
            'mean_speed': float(speed[m].mean()) if m.any() else None,
            'max_speed': float(speed[m].max()) if m.any() else None,
            'distance': float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0,
            'min_clearance': finite(np.nanmin(clear[m])) if m.any() and np.isfinite(clear[m]).any() else None,
            'mean_gate_offset': float(np.mean(offs)) if offs else None,
            'max_gate_offset': float(np.max(offs)) if offs else None,
        })
        t_prev = t_end
    return rows


def gate_table(run: RunData) -> List[dict]:
    rows = []
    passes = run.summary.get('passes', [])
    events = run.summary.get('events', [])
    for gid in run.course.order:
        ps = [x for x in passes if x['gate_id'] == gid]
        offs = np.array([[x['offset_y'], x['offset_z']] for x in ps]) if ps else np.zeros((0, 2))
        clears = [x['clearance'] for x in ps if finite(x.get('clearance')) is not None]
        rows.append({
            'gate_id': gid,
            'passes': len(ps),
            'hits': sum(1 for e in events if e['type'] == 'HIT' and e['gate_id'] == gid),
            'missed': sum(1 for e in events if e['type'] == 'MISSED' and e['gate_id'] == gid),
            'mean_speed': float(np.mean([x['speed'] for x in ps])) if ps else None,
            'mean_offset_y': float(offs[:, 0].mean()) if len(offs) else None,
            'mean_offset_z': float(offs[:, 1].mean()) if len(offs) else None,
            'max_offset': float(np.hypot(offs[:, 0], offs[:, 1]).max()) if len(offs) else None,
            'std_offset_y': float(offs[:, 0].std()) if len(offs) > 1 else None,
            'std_offset_z': float(offs[:, 1].std()) if len(offs) > 1 else None,
            'min_clearance': float(min(clears)) if clears else None,
            'mean_approach_angle': float(np.mean([x.get('approach_angle', np.nan) for x in ps]))
            if ps else None,
        })
    return rows


def phase_table(run: RunData) -> List[dict]:
    """Time spent in each race-manager phase after the start command (from telemetry)."""
    log = run.race_log
    if not log or 'state' not in log:
        return []
    rows = []
    entries = [(float(t) - run.start, str(s)) for t, src, s in
               zip(log['rx'], log['source'], log['state']) if str(src) == 'race_manager']
    entries = [e for e in entries if e[0] >= -1e-3]
    end = run.summary.get('elapsed_time')
    for k, (t0, state) in enumerate(entries):
        if state in ('FINISHED', 'ABORTED'):
            break
        t1 = entries[k + 1][0] if k + 1 < len(entries) else (end if end is not None else t0)
        if state == 'LANDING' and end is not None:
            t1 = end                      # the referee stops the clock at touchdown
        rows.append({'phase': state, 'start': t0, 'duration': max(0.0, t1 - t0)})
    return rows


def split_matrix(run: RunData) -> np.ndarray:
    """(laps, gates) time from the previous pass (or the start) to each gate pass; NaN if missing."""
    gpl = run.course.gates_per_lap
    out = np.full((run.course.laps, gpl), np.nan)
    prev = 0.0
    for ps in sorted(run.summary.get('passes', []), key=lambda x: x['index']):
        lap, k = divmod(ps['index'], gpl)
        out[lap, k] = ps['race_time'] - prev
        prev = ps['race_time']
    return out


def tracking(run: RunData) -> Optional[dict]:
    """Commanded reference (control manager) vs ground truth, in the world frame."""
    c = run.control
    t_wo = run.alignment()
    if not c or t_wo is None or 'ref_x' not in c or not len(run.traj):
        return None
    tc = run.race_time(c['t'])
    ref = transform_points(t_wo, np.stack([c['ref_x'], c['ref_y'], c['ref_z']], axis=1))
    t, p = run.t, run.p
    m = (tc >= max(t[0], 0.0)) & (tc <= t[-1])
    if m.sum() < 10:
        return None
    tc, ref = tc[m], ref[m]
    act = np.stack([np.interp(tc, t, p[:, k]) for k in range(3)], axis=1)
    err = act - ref
    norm = np.linalg.norm(err, axis=1)
    flying = c['is_fly'][m] > 0.5 if 'is_fly' in c else np.ones(len(tc), bool)
    sel = flying if flying.any() else np.ones(len(tc), bool)
    return {
        't': tc, 'ref': ref, 'act': act, 'err': err, 'norm': norm, 'flying': flying,
        'rmse': float(np.sqrt(np.mean(norm[sel] ** 2))),
        'max': float(norm[sel].max()),
        'p95': float(np.percentile(norm[sel], 95)),
        'rmse_xy': float(np.sqrt(np.mean(np.sum(err[sel, :2] ** 2, axis=1)))),
        'rmse_z': float(np.sqrt(np.mean(err[sel, 2] ** 2))),
    }


def compute_metrics(run: RunData) -> Dict[str, object]:
    s = run.summary
    t, p, v = run.t, run.p, run.v
    m = race_window(run)
    speed = np.linalg.norm(v, axis=1)
    acc = smooth_derivative(t, v, 2) if len(t) > 5 else np.zeros_like(v)
    acc_n = np.linalg.norm(acc, axis=1)
    att = attitude_deg(run) if len(t) else np.zeros((0, 3))
    tilt = np.hypot(att[:, 0], att[:, 1]) if len(att) else np.zeros(0)
    rest = s.get('rest_height') or (p[0, 2] if len(p) else 0.0)
    airborne = (p[:, 2] > rest + 0.05) & m if len(p) else np.zeros(0, bool)
    dt = np.diff(t, prepend=t[0]) if len(t) else np.zeros(0)
    out: Dict[str, object] = {
        'run': run.name,
        'valid': bool(s.get('valid')),
        'state': s.get('state'),
        'reason': s.get('reason'),
        'elapsed_time': finite(s.get('elapsed_time')),
        'adjusted_time': finite(s.get('adjusted_time')),
        'penalty_time': finite(s.get('penalty_time')),
        'hits': s.get('hits'),
        'missed_entries': s.get('missed_entries'),
        'wrong_way': s.get('wrong_way'),
        'gates_passed': s.get('gates_passed'),
        'gates_total': s.get('gates_total'),
        'laps_completed': s.get('laps_completed'),
        'distance_flown': finite(s.get('distance_flown')),
        'max_speed': float(speed[m].max()) if m.any() else None,
        'mean_speed': float(speed[m].mean()) if m.any() else None,
        'p95_speed': float(np.percentile(speed[m], 95)) if m.any() else None,
        'max_accel': float(acc_n[m].max()) if m.any() else None,
        'p95_accel': float(np.percentile(acc_n[m], 95)) if m.any() else None,
        'max_tilt_deg': float(tilt[m].max()) if m.any() else None,
        'airborne_time': float(dt[airborne].sum()) if len(dt) else None,
        'max_height': float(p[m, 2].max()) if m.any() else None,
        'min_clearance': finite(s.get('min_clearance')),
    }
    laps = lap_table(run)
    if laps:
        lt = [r['lap_time'] for r in laps]
        out.update({'best_lap': min(lt), 'mean_lap': float(np.mean(lt)),
                    'lap_std': float(np.std(lt))})
        fly = [r['flying_lap_time'] for r in laps if r['flying_lap_time'] is not None]
        if fly:
            out.update({'best_flying_lap': min(fly), 'mean_flying_lap': float(np.mean(fly))})
    passes = s.get('passes', [])
    if passes:
        offs = np.array([math.hypot(x['offset_y'], x['offset_z']) for x in passes])
        out.update({'mean_gate_offset': float(offs.mean()), 'max_gate_offset': float(offs.max()),
                    'mean_gate_speed': float(np.mean([x['speed'] for x in passes]))})
        angles = [x['approach_angle'] for x in passes if finite(x.get('approach_angle')) is not None]
        if angles:
            out.update({'mean_approach_angle_deg': float(np.mean(angles)),
                        'max_approach_angle_deg': float(np.max(angles))})
    for row in phase_table(run):
        out[f"phase_{row['phase'].lower()}_s"] = row['duration']
    tr = tracking(run)
    if tr is not None:
        out.update({'tracking_rmse': tr['rmse'], 'tracking_max': tr['max'],
                    'tracking_p95': tr['p95'], 'tracking_rmse_xy': tr['rmse_xy'],
                    'tracking_rmse_z': tr['rmse_z']})
    c = run.control
    if c and 'ocp_ms' in c:
        ocp = c['ocp_ms'][np.isfinite(c['ocp_ms'])]
        if len(ocp):
            out.update({'ocp_ms_mean': float(ocp.mean()), 'ocp_ms_p95': float(np.percentile(ocp, 95)),
                        'ocp_ms_max': float(ocp.max())})
        it = c.get('iteration_ms')
        if it is not None and np.isfinite(it).any():
            out['control_iteration_ms_p95'] = float(np.nanpercentile(it, 95))
        if all(k in c for k in ('u0', 'u1', 'u2', 'u3')):
            u = np.stack([c[f'u{k}'] for k in range(4)], axis=1)
            tc = run.race_time(c['t'])
            mm = (tc >= 0) & np.isfinite(u).all(axis=1)
            if mm.any():
                total = u[mm].sum(axis=1)
                dtc = np.diff(tc[mm], prepend=tc[mm][0])
                out.update({'mean_total_thrust_N': float(total.mean()),
                            'thrust_impulse_Ns': float((total * dtc).sum())})
    mo = run.motors
    if mo and 'm0' in mo:
        u = np.stack([mo[f'm{k}'] for k in range(4)], axis=1)
        tm = run.race_time(mo['rx'])
        mm = (tm >= 0) & np.isfinite(u).all(axis=1)
        if mm.any():
            out['motor_saturation_pct'] = float(100.0 * np.mean((u[mm] >= 0.99).any(axis=1) |
                                                                (u[mm] <= 0.01).any(axis=1)))
            out['motor_mean_cmd'] = float(u[mm].mean())
    if run.plan and run.plan.get('prediction'):
        pred = run.plan['prediction']
        out['planned_duration'] = finite(pred.get('duration'))
        out['planning_ms'] = finite(pred.get('planning_ms'))
        ms = run.mission_start()
        if ms is not None and out.get('planned_duration') is not None:
            done = None
            pl = run.planning
            if pl and 'state' in pl:
                completed = np.flatnonzero(pl['state'] == 3)
                if len(completed):
                    done = float(pl['t'][completed[-1]]) - run.start
            if done is not None:
                out['mission_duration_actual'] = done - ms
                out['mission_delay'] = done - ms - out['planned_duration']
    if run.plan and run.plan.get('validation'):
        out['plan_min_clearance'] = finite(run.plan['validation'].get('min_clearance'))
        out['plan_min_wall_margin'] = finite(run.plan['validation'].get('min_wall_margin'))
    return out
