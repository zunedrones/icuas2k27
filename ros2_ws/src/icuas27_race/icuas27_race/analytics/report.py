"""Race report: `ros2 run icuas27_race race_report <run_dir>` writes <run_dir>/report/report.html
(self-contained: figures embedded, interactive replay, tables) plus metrics.json and PNGs."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .metrics import (attitude_deg, compute_metrics, gate_table, lap_table, phase_table,  # noqa: E402
                      race_window,
                      smooth_derivative, split_matrix, tracking)
from .rundata import RunData, load_run  # noqa: E402

EVENT_STYLE = {'PASS': ('o', '#2ca02c'), 'HIT': ('x', '#d62728'), 'MISSED': ('s', '#ff7f0e'),
               'WRONG_WAY': ('v', '#9467bd'), 'OUT_OF_ORDER': ('d', '#8c564b'),
               'OUT_OF_BOUNDS': ('X', '#d62728'), 'CRASH': ('X', '#d62728'),
               'FINISH': ('*', '#1f77b4'), 'TIMEOUT': ('X', '#d62728')}


# ----------------------------------------------------------------------------- utils
def fig_to_b64(fig, path: Optional[str] = None) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    if path:
        with open(path, 'wb') as fh:
            fh.write(buf.getvalue())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def fmt(x, nd=3, unit=''):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '–'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, str):
        return html.escape(x)
    if isinstance(x, (int, np.integer)):
        return f'{x}{unit}'
    return f'{x:.{nd}f}{unit}'


def gate_segment_xy(gate, outer=True):
    w = 0.5 * (gate.outer_width if outer else gate.inner_width)
    left = gate.rotation[:, 1]
    a = gate.center + w * left
    b = gate.center - w * left
    return a, b


def draw_course_xy(ax, run: RunData, labels=True):
    c = run.course
    lo, hi = c.arena_min, c.arena_max
    ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, lw=1.5,
                           ec='#555555'))
    ax.add_patch(Rectangle((c.pad_center[0] - c.pad_size[0] / 2, c.pad_center[1] - c.pad_size[1] / 2),
                           c.pad_size[0], c.pad_size[1], color='#c49a6c', alpha=0.5))
    for g in c.gates.values():
        a, b = gate_segment_xy(g, outer=True)
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#333333', lw=4, solid_capstyle='butt', zorder=3)
        a, b = gate_segment_xy(g, outer=False)
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#bbbbbb', lw=2, solid_capstyle='butt', zorder=4)
        ax.arrow(g.center[0] - 0.25 * g.normal[0], g.center[1] - 0.25 * g.normal[1],
                 0.4 * g.normal[0], 0.4 * g.normal[1], head_width=0.08, color='#777777', zorder=2)
        if labels:
            ax.text(g.center[0] + 0.25, g.center[1] + 0.25, f'G{g.gate_id}\nz={g.center[2]:.2f}',
                    fontsize=8, color='#333333')
    ax.set_aspect('equal')
    ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
    ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    ax.set_xlabel('x [m] (world, east)')
    ax.set_ylabel('y [m] (world, north)')


def speed_collection(xy: np.ndarray, speed: np.ndarray, vmax: float, lw=2.0):
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(seg, cmap='turbo', norm=plt.Normalize(0, vmax), lw=lw)
    lc.set_array(speed[:-1])
    return lc


def mark_passes(ax, run: RunData, laps_only=False):
    passes = run.summary.get('passes', [])
    gpl = run.course.gates_per_lap
    for ps in passes:
        last = ps['index'] % gpl == gpl - 1
        if laps_only and not last:
            continue
        ax.axvline(ps['race_time'], color='#2ca02c' if not last else '#1f77b4',
                   lw=0.6 if not last else 1.2, alpha=0.35 if not last else 0.7)


# ----------------------------------------------------------------------------- figures
def fig_topdown(run: RunData, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_course_xy(ax, run)
    p, v = run.p, run.v
    speed = np.linalg.norm(v, axis=1)
    vmax = max(float(speed.max()) if len(speed) else 1.0, 0.5)
    if run.plan and run.plan.get('prediction'):
        pp = np.asarray(run.plan['prediction']['p'])
        ax.plot(pp[:, 0], pp[:, 1], '--', color='#888888', lw=0.8, label='planned (PMM prediction)')
    m = race_window(run)
    if m.sum() > 2:
        lc = speed_collection(p[m][:, :2], speed[m], vmax)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('speed [m/s]')
    for ev in run.summary.get('events', []):
        mk, col = EVENT_STYLE.get(ev['type'], ('.', 'k'))
        if ev['type'] == 'PASS':
            continue
        ax.plot(ev['position'][0], ev['position'][1], mk, color=col, ms=9, mew=2,
                label=ev['type'])
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc='upper left', fontsize=8)
    ax.set_title(f'Top view - flown path coloured by speed ({run.name})')
    return fig_to_b64(fig, os.path.join(out_dir, 'topdown.png'))


def fig_3d(run: RunData, out_dir: str) -> str:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    m = race_window(run)
    p = run.p[m]
    speed = np.linalg.norm(run.v[m], axis=1)
    if len(p) > 2:
        ax.scatter(p[::3, 0], p[::3, 1], p[::3, 2], c=speed[::3], cmap='turbo', s=2)
    for g in run.course.gates.values():
        for corners, col, lw in ((g.corners(False), '#333333', 2.5), (g.corners(True), '#999999', 1.2)):
            loop = np.vstack([corners, corners[:1]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=col, lw=lw)
        ax.text(g.center[0], g.center[1], g.center[2] + 0.4, f'G{g.gate_id}', fontsize=8)
        a, b = gate_segment_xy(g, outer=False)
        for e in (a, b):
            ax.plot([e[0], e[0]], [e[1], e[1]], [0, g.center[2] - 0.3], color='#aaaaaa', lw=1)
    lo, hi = run.course.arena_min, run.course.arena_max
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(0, hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.view_init(elev=28, azim=-120)
    ax.set_title('3D flown path (colour = speed)')
    return fig_to_b64(fig, os.path.join(out_dir, '3d.png'))


def fig_profiles(run: RunData, out_dir: str) -> str:
    t, p, v = run.t, run.p, run.v
    m = race_window(run)
    speed = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(smooth_derivative(t, v, 2), axis=1) if len(t) > 5 else np.zeros(len(t))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax = axes[0]
    ax.plot(t[m], speed[m], color='#1f77b4', lw=1.2, label='actual (ground truth)')
    ms = run.mission_start()
    if run.plan and run.plan.get('prediction') and ms is not None:
        pr = run.plan['prediction']
        tp = np.asarray(pr['t']) + ms
        sp = np.linalg.norm(np.asarray(pr['v']), axis=1)
        ax.plot(tp, sp, '--', color='#ff7f0e', lw=1.0, label='planned (PMM)')
    mark_passes(ax, run)
    ax.set_ylabel('speed [m/s]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t[m], p[m, 2], color='#2ca02c', lw=1.2, label='z')
    for ps in run.summary.get('passes', []):
        ax.plot(ps['race_time'], ps['position'][2], 'o', color='#2ca02c', ms=3)
    mark_passes(ax, run, laps_only=True)
    ax.set_ylabel('height [m]')
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(t[m], acc[m], color='#d62728', lw=1.0, label='|a| (from ground-truth velocity)')
    if run.plan and run.plan.get('prediction') and ms is not None:
        pr = run.plan['prediction']
        ax.plot(np.asarray(pr['t']) + ms, np.linalg.norm(np.asarray(pr['a']), axis=1), '--',
                color='#ff7f0e', lw=0.9, label='planned |a|')
    mark_passes(ax, run, laps_only=True)
    ax.set_ylabel('acceleration [m/s²]')
    ax.set_xlabel('race time [s] (0 = start command)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle('Speed, height and acceleration (green lines: gate passes, blue: lap ends)')
    return fig_to_b64(fig, os.path.join(out_dir, 'profiles.png'))


def fig_attitude(run: RunData, out_dir: str) -> str:
    t = run.t
    m = race_window(run)
    att = attitude_deg(run)
    w = np.stack([run.traj['wx'], run.traj['wy'], run.traj['wz']], axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for k, lab in enumerate(('roll', 'pitch', 'yaw')):
        axes[0].plot(t[m], att[m, k], lw=1, label=lab)
    axes[0].set_ylabel('attitude [deg]')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(alpha=0.3)
    for k, lab in enumerate(('p (roll rate)', 'q (pitch rate)', 'r (yaw rate)')):
        axes[1].plot(t[m], w[m, k], lw=0.8, label=lab)
    axes[1].set_ylabel('body rates [rad/s]')
    axes[1].set_xlabel('race time [s]')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(alpha=0.3)
    fig.suptitle('Attitude and body rates (ground truth)')
    return fig_to_b64(fig, os.path.join(out_dir, 'attitude.png'))


def fig_laps(run: RunData, out_dir: str) -> Optional[str]:
    rows = lap_table(run)
    if not rows:
        return None
    splits = split_matrix(run)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={'width_ratios': [1, 1.3]})
    ax = axes[0]
    laps = [r['lap'] for r in rows]
    ax.bar(np.array(laps) - 0.2, [r['lap_time'] for r in rows], width=0.4, label='lap (to last gate)')
    fly = [r['flying_lap_time'] if r['flying_lap_time'] is not None else np.nan for r in rows]
    ax.bar(np.array(laps) + 0.2, fly, width=0.4, label='flying lap (G1 to G1)')
    for x, r in zip(laps, rows):
        ax.text(x - 0.2, r['lap_time'] + 0.05, f"{r['lap_time']:.2f}", ha='center', fontsize=7)
    ax.set_xlabel('lap')
    ax.set_ylabel('time [s]')
    ax.set_ylim(0, 1.25 * max(r['lap_time'] for r in rows))
    ax.legend(fontsize=8, loc='upper center', ncol=2)
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('Lap times')
    ax = axes[1]
    im = ax.imshow(splits, cmap='viridis', aspect='auto')
    ax.set_xticks(range(run.course.gates_per_lap))
    ax.set_xticklabels([f'G{g}' for g in run.course.order])
    ax.set_yticks(range(run.course.laps))
    ax.set_yticklabels([f'L{k + 1}' for k in range(run.course.laps)])
    for i in range(splits.shape[0]):
        for j in range(splits.shape[1]):
            if np.isfinite(splits[i, j]):
                ax.text(j, i, f'{splits[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white')
    fig.colorbar(im, ax=ax, fraction=0.04).set_label('split [s]')
    ax.set_title('Split time to each gate (from the previous gate)')
    return fig_to_b64(fig, os.path.join(out_dir, 'laps.png'))


def fig_gate_offsets(run: RunData, out_dir: str, hull: float) -> Optional[str]:
    passes = run.summary.get('passes', [])
    events = run.summary.get('events', [])
    if not passes and not events:
        return None
    gates = [run.course.gates[g] for g in run.course.order]
    n = len(gates)
    cols = min(n, 4)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows), squeeze=False)
    for k, g in enumerate(gates):
        ax = axes[k // cols][k % cols]
        wi, hi = g.inner_width / 2, g.inner_height / 2
        wo, ho = g.outer_width / 2, g.outer_height / 2
        ax.add_patch(Rectangle((-wo, -ho), 2 * wo, 2 * ho, color='#444444', alpha=0.8))
        ax.add_patch(Rectangle((-wi, -hi), 2 * wi, 2 * hi, color='white'))
        ax.add_patch(Rectangle((-wi + hull, -hi + hull), 2 * (wi - hull), 2 * (hi - hull),
                               fill=False, ls='--', ec='#2ca02c', lw=1))
        ps = [x for x in passes if x['gate_id'] == g.gate_id]
        if ps:
            ax.scatter([-x['offset_y'] for x in ps], [x['offset_z'] for x in ps],
                            c=[x['lap'] for x in ps], cmap='plasma', vmin=1,
                            vmax=run.course.laps, s=22, zorder=5)
        for e in events:
            if e['gate_id'] == g.gate_id and e['type'] in ('MISSED', 'WRONG_WAY') and \
                    math.isfinite(e.get('offset_y', math.nan)):
                ax.plot(-e['offset_y'], e['offset_z'], 'x', color='#d62728', ms=8, mew=2)
        ax.set_xlim(-wo - 0.1, wo + 0.1)
        ax.set_ylim(-ho - 0.1, ho + 0.1)
        ax.set_aspect('equal')
        ax.set_title(f'G{g.gate_id} ({len(ps)} passes)', fontsize=9)
        ax.tick_params(labelsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle('Crossing points seen from the entry side (colour = lap; dashed = hull-safe area)')
    return fig_to_b64(fig, os.path.join(out_dir, 'gate_offsets.png'))


def fig_clearance(run: RunData, out_dir: str) -> Optional[str]:
    c = run.traj.get('clearance')
    if c is None or not np.isfinite(c).any():
        return None
    t = run.t
    m = race_window(run) & np.isfinite(c)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(t[m], np.clip(c[m], -0.1, 0.6) * 100, lw=0.8, color='#1f77b4')
    ax.axhline(0, color='#d62728', lw=1, label='contact')
    ax.axhline(5, color='#ff7f0e', lw=0.8, ls='--', label='5 cm')
    for ps in run.summary.get('passes', []):
        if ps.get('clearance') is not None and math.isfinite(ps['clearance']):
            ax.plot(ps['race_time'], ps['clearance'] * 100, 'o', color='#2ca02c', ms=3)
    ax.set_ylabel('hull clearance [cm] (clipped at 60)')
    ax.set_xlabel('race time [s]')
    ax.set_title('Distance between the drone hull and the closest gate structure '
                 '(only within 0.4 m of a gate)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig_to_b64(fig, os.path.join(out_dir, 'clearance.png'))


def fig_tracking(run: RunData, out_dir: str) -> Optional[str]:
    tr = tracking(run)
    if tr is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]
    ax.plot(tr['t'], tr['norm'] * 100, lw=0.8, label='|error|')
    ax.plot(tr['t'], np.linalg.norm(tr['err'][:, :2], axis=1) * 100, lw=0.6, alpha=0.7, label='horizontal')
    ax.plot(tr['t'], np.abs(tr['err'][:, 2]) * 100, lw=0.6, alpha=0.7, label='vertical')
    mark_passes(ax, run, laps_only=True)
    ax.set_xlabel('race time [s]')
    ax.set_ylabel('tracking error [cm]')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"Commanded reference vs ground truth (RMSE {tr['rmse'] * 100:.1f} cm)")
    ax = axes[1]
    sel = tr['flying']
    ax.hist(tr['norm'][sel] * 100, bins=40, color='#1f77b4')
    ax.set_xlabel('|error| [cm] (in flight)')
    ax.set_ylabel('samples')
    ax.grid(alpha=0.3)
    return fig_to_b64(fig, os.path.join(out_dir, 'tracking.png'))


def fig_actuation(run: RunData, out_dir: str) -> Optional[str]:
    mo, c = run.motors, run.control
    if not mo and not c:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    if mo and 'm0' in mo:
        tm = run.race_time(mo['rx'])
        m = tm >= -1.0
        for k in range(4):
            ax.plot(tm[m], mo[f'm{k}'][m], lw=0.6, label=f'motor {k}')
        ax.axhline(1.0, color='#d62728', lw=0.8, ls='--')
        ax.set_ylabel('normalised command')
        ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title('Motor commands sent to PX4')
    ax = axes[1]
    if c and 'ocp_ms' in c:
        tc = run.race_time(c['t'])
        m = tc >= -1.0
        ax.plot(tc[m], c['ocp_ms'][m], lw=0.6, label='NMPC solve [ms]')
        if 'iteration_ms' in c:
            ax.plot(tc[m], c['iteration_ms'][m], lw=0.6, label='control iteration [ms]')
        ax.axhline(10.0, color='#d62728', lw=0.8, ls='--', label='100 Hz budget')
        ax.legend(fontsize=7, loc='upper right')
    ax.set_ylabel('time [ms]')
    ax.set_xlabel('race time [s]')
    ax.grid(alpha=0.3)
    return fig_to_b64(fig, os.path.join(out_dir, 'actuation.png'))


def fig_speed_hist(run: RunData, out_dir: str) -> str:
    m = race_window(run)
    speed = np.linalg.norm(run.v[m], axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(speed, bins=40, color='#1f77b4')
    ax.set_xlabel('speed [m/s]')
    ax.set_ylabel('samples (100 Hz)')
    ax.set_title('Speed distribution')
    ax.grid(alpha=0.3)
    return fig_to_b64(fig, os.path.join(out_dir, 'speed_hist.png'))


# ----------------------------------------------------------------------------- html
def table(rows: List[dict], columns: List[tuple]) -> str:
    head = ''.join(f'<th>{html.escape(lab)}</th>' for _, lab, _ in columns)
    body = []
    for r in rows:
        cells = ''.join(f'<td>{fmt(r.get(key), nd)}</td>' for key, _, nd in columns)
        body.append(f'<tr>{cells}</tr>')
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def replay_payload(run: RunData) -> dict:
    t, p, v = run.t, run.p, run.v
    keep = t >= -1.0
    t, p, v = t[keep], p[keep], v[keep]
    if len(t) > 1:
        step = max(1, int(round(0.033 / max(1e-3, float(np.median(np.diff(t)))))))
        t, p, v = t[::step], p[::step], v[::step]
    c = run.course
    gates = []
    for gid in c.order:
        g = c.gates[gid]
        a, b = gate_segment_xy(g, outer=True)
        gates.append({'id': gid, 'a': a[:2].round(3).tolist(), 'b': b[:2].round(3).tolist(),
                      'c': g.center.round(3).tolist(), 'h': g.outer_height})
    return {
        't': np.round(t, 3).tolist(),
        'x': np.round(p[:, 0], 3).tolist(), 'y': np.round(p[:, 1], 3).tolist(),
        'z': np.round(p[:, 2], 3).tolist(),
        's': np.round(np.linalg.norm(v, axis=1), 2).tolist(),
        'arena': [c.arena_min.tolist(), c.arena_max.tolist()],
        'pad': [c.pad_center.tolist(), c.pad_size.tolist()],
        'gates': gates,
        'sequence': c.sequence,
        'gpl': c.gates_per_lap,
        'passes': [[round(x['race_time'], 3), x['index']] for x in run.summary.get('passes', [])],
        'events': [[round(e['race_time'], 3), e['type'], round(e['position'][0], 3),
                    round(e['position'][1], 3), round(e['position'][2], 3)]
                   for e in run.summary.get('events', []) if e['type'] != 'PASS'],
        'end': run.summary.get('elapsed_time') or (float(t[-1]) if len(t) else 0.0),
    }


REPLAY_JS = r"""
(function(){
const D = REPLAY_DATA;
const top = document.getElementById('rp-top'), side = document.getElementById('rp-side');
const ctx = top.getContext('2d'), sctx = side.getContext('2d');
const slider = document.getElementById('rp-slider'), hud = document.getElementById('rp-hud');
const btn = document.getElementById('rp-play'), rate = document.getElementById('rp-rate');
const N = D.t.length; if (!N) { hud.textContent = 'no trajectory'; return; }
let idx = 0, playing = false, last = null, simT = D.t[0];
slider.max = N - 1;
const lo = D.arena[0], hi = D.arena[1], m = 0.4;
function sx(x){ return (x - lo[0] + m) / (hi[0] - lo[0] + 2*m) * top.width; }
function sy(y){ return top.height - (y - lo[1] + m) / (hi[1] - lo[1] + 2*m) * top.height; }
const slen = []; let acc = 0; for (let i=0;i<N;i++){ if(i) acc += Math.hypot(D.x[i]-D.x[i-1], D.y[i]-D.y[i-1]); slen.push(acc); }
function vx(i){ return 20 + (D.t[i]-D.t[0]) / Math.max(1e-6, D.t[N-1]-D.t[0]) * (side.width-40); }
function vz(z){ return side.height - 15 - z / hi[2] * (side.height - 30); }
function color(s){ const u = Math.min(1, s/4); const r = Math.round(255*Math.min(1, 2*u)), b = Math.round(255*Math.min(1, 2*(1-u))); return `rgb(${r},${Math.round(120+80*(1-Math.abs(u-0.5)*2))},${b})`; }
function passedAt(t){ let n=0; for (const p of D.passes) if (p[0] <= t) n++; return n; }
function draw(){
  const t = D.t[idx];
  ctx.clearRect(0,0,top.width,top.height);
  ctx.strokeStyle = '#888'; ctx.lineWidth = 1.5;
  ctx.strokeRect(sx(lo[0]), sy(hi[1]), sx(hi[0])-sx(lo[0]), sy(lo[1])-sy(hi[1]));
  const pc = D.pad[0], ps = D.pad[1];
  ctx.fillStyle = 'rgba(196,154,108,0.6)';
  ctx.fillRect(sx(pc[0]-ps[0]/2), sy(pc[1]+ps[1]/2), sx(pc[0]+ps[0]/2)-sx(pc[0]-ps[0]/2), sy(pc[1]-ps[1]/2)-sy(pc[1]+ps[1]/2));
  const np = passedAt(t), next = np < D.sequence.length ? D.sequence[np] : -1;
  for (const g of D.gates){
    ctx.strokeStyle = g.id === next ? '#f5b800' : '#444'; ctx.lineWidth = g.id === next ? 6 : 4;
    ctx.beginPath(); ctx.moveTo(sx(g.a[0]), sy(g.a[1])); ctx.lineTo(sx(g.b[0]), sy(g.b[1])); ctx.stroke();
    ctx.fillStyle = '#333'; ctx.font = '11px sans-serif'; ctx.fillText('G'+g.id, sx(g.c[0])+8, sy(g.c[1])-8);
  }
  const tail = 3.0;
  for (let i=1;i<=idx;i++){
    if (D.t[i] < t - tail) continue;
    ctx.strokeStyle = color(D.s[i]); ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(sx(D.x[i-1]), sy(D.y[i-1])); ctx.lineTo(sx(D.x[i]), sy(D.y[i])); ctx.stroke();
  }
  for (const e of D.events){ if (e[0] > t) continue; ctx.fillStyle = '#d62728'; ctx.fillRect(sx(e[2])-4, sy(e[3])-4, 8, 8); }
  ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(sx(D.x[idx]), sy(D.y[idx]), 6, 0, 2*Math.PI); ctx.fill();
  sctx.clearRect(0,0,side.width,side.height);
  sctx.strokeStyle = '#ccc'; sctx.beginPath(); sctx.moveTo(20, vz(0)); sctx.lineTo(side.width-20, vz(0)); sctx.stroke();
  sctx.strokeStyle = '#1f77b4'; sctx.lineWidth = 1; sctx.beginPath();
  for (let i=0;i<N;i++){ const X = vx(i), Y = vz(D.z[i]); if (i) sctx.lineTo(X,Y); else sctx.moveTo(X,Y); }
  sctx.stroke();
  sctx.fillStyle = '#000'; sctx.beginPath(); sctx.arc(vx(idx), vz(D.z[idx]), 4, 0, 2*Math.PI); sctx.fill();
  sctx.fillStyle = '#555'; sctx.font = '11px sans-serif'; sctx.fillText('height vs time', 24, 14);
  const lap = Math.min(Math.floor(np / D.gpl) + 1, D.sequence.length / D.gpl);
  hud.textContent = `t = ${t.toFixed(2)} s   lap ${lap}/${D.sequence.length / D.gpl}   gates ${np}/${D.sequence.length}   next ${next > 0 ? 'G'+next : 'land'}   v = ${D.s[idx].toFixed(2)} m/s   z = ${D.z[idx].toFixed(2)} m`;
  slider.value = idx;
}
function frame(ts){
  if (!playing){ last = null; return; }
  if (last !== null){ simT += (ts - last) / 1000 * parseFloat(rate.value); }
  last = ts;
  while (idx < N-1 && D.t[idx+1] <= simT) idx++;
  if (idx >= N-1){ playing = false; btn.textContent = '▶ play'; }
  draw();
  if (playing) requestAnimationFrame(frame);
}
btn.onclick = () => { playing = !playing; btn.textContent = playing ? '❚❚ pause' : '▶ play';
  if (playing){ if (idx >= N-1){ idx = 0; } simT = D.t[idx]; requestAnimationFrame(frame); } };
slider.oninput = () => { idx = parseInt(slider.value); simT = D.t[idx]; draw(); };
draw();
})();
"""

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2330}
header{background:#1d2330;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:22px} header p{margin:4px 0 0;color:#c8cdd8;font-size:13px}
main{padding:18px 28px;max-width:1280px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 18px}
.card{background:#fff;border-radius:8px;padding:12px 16px;min-width:150px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card .k{font-size:12px;color:#5b6475} .card .v{font-size:22px;font-weight:600;margin-top:2px}
.ok{color:#1a7f37} .bad{color:#c62828}
section{background:#fff;border-radius:8px;padding:14px 18px;margin:0 0 18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{font-size:17px;margin:2px 0 10px}
img{max-width:100%;height:auto}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0}
th,td{border-bottom:1px solid #e3e6eb;padding:4px 9px;text-align:right}
th{background:#f0f2f5;text-align:center}
td:first-child,th:first-child{text-align:left}
.replay canvas{background:#fbfbfc;border:1px solid #dde1e7;border-radius:6px}
.controls{display:flex;gap:10px;align-items:center;margin:8px 0}
#rp-slider{flex:1} #rp-hud{font-family:ui-monospace,monospace;font-size:13px;margin:6px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
small{color:#5b6475}
"""


def build_html(run: RunData, metrics: dict, figs: dict) -> str:
    s = run.summary
    valid = bool(s.get('valid'))
    status = 'VALID' if valid else f"INVALID ({s.get('state')})"
    cards = [
        ('Result', f'<span class="{"ok" if valid else "bad"}">{html.escape(status)}</span>'),
        ('T elapsed', fmt(metrics.get('elapsed_time'), 3, ' s')),
        ('Penalties', f"{fmt(metrics.get('penalty_time'), 1, ' s')} "
                      f"<small>({s.get('hits', 0)} hits, {s.get('missed_entries', 0)} missed)</small>"),
        ('T adjusted', fmt(metrics.get('adjusted_time'), 3, ' s')),
        ('Gates', f"{s.get('gates_passed', 0)}/{s.get('gates_total', 0)}"),
        ('Laps', f"{s.get('laps_completed', 0)}/{s.get('laps_total', 0)}"),
        ('Best lap', fmt(metrics.get('best_lap'), 3, ' s')),
        ('Max speed', fmt(metrics.get('max_speed'), 2, ' m/s')),
        ('Mean speed', fmt(metrics.get('mean_speed'), 2, ' m/s')),
        ('Distance', fmt(metrics.get('distance_flown'), 1, ' m')),
        ('Min clearance', fmt((metrics.get('min_clearance') or math.nan) * 100, 1, ' cm')),
        ('Tracking RMSE', fmt((metrics.get('tracking_rmse') or math.nan) * 100, 1, ' cm')),
    ]
    cards_html = ''.join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
                         for k, v in cards)
    reason = f"<p><b>Reason:</b> {html.escape(str(s.get('reason')))}</p>" if s.get('reason') else ''
    lap_cols = [('lap', 'Lap', 0), ('lap_time', 'Lap time [s]', 3),
                ('flying_lap_time', 'Flying lap [s]', 3), ('mean_speed', 'Mean speed [m/s]', 2),
                ('max_speed', 'Max speed [m/s]', 2), ('distance', 'Distance [m]', 2),
                ('min_clearance', 'Min clearance [m]', 3), ('mean_gate_offset', 'Mean offset [m]', 3),
                ('max_gate_offset', 'Max offset [m]', 3)]
    gate_cols = [('gate_id', 'Gate', 0), ('passes', 'Passes', 0), ('hits', 'Hits', 0),
                 ('missed', 'Missed', 0), ('mean_speed', 'Speed [m/s]', 2),
                 ('mean_offset_y', 'Mean y [m]', 3), ('mean_offset_z', 'Mean z [m]', 3),
                 ('std_offset_y', 'Std y [m]', 3), ('std_offset_z', 'Std z [m]', 3),
                 ('max_offset', 'Max |offset| [m]', 3), ('min_clearance', 'Min clearance [m]', 3),
                 ('mean_approach_angle', 'Approach angle [deg]', 1)]
    phase_cols = [('phase', 'Phase', 0), ('start', 'Start [s]', 2), ('duration', 'Duration [s]', 2)]
    ev_rows = [{'t': e['race_time'], 'type': e['type'], 'gate': e['gate_id'], 'idx': e['gate_index'],
                'lap': e['lap'], 'oy': e['offset_y'], 'oz': e['offset_z'], 'v': e['speed'],
                'c': e['clearance'], 'detail': e['detail']} for e in s.get('events', [])]
    ev_head = ('<tr><th>t [s]</th><th>event</th><th>gate</th><th>#</th><th>lap</th><th>y [m]</th>'
               '<th>z [m]</th><th>v [m/s]</th><th>clear [m]</th><th>detail</th></tr>')
    ev_body = ''.join(
        f"<tr><td>{fmt(r['t'])}</td><td>{r['type']}</td><td>{r['gate']}</td><td>{r['idx']}</td>"
        f"<td>{r['lap']}</td><td>{fmt(r['oy'])}</td><td>{fmt(r['oz'])}</td><td>{fmt(r['v'], 2)}</td>"
        f"<td>{fmt(r['c'])}</td><td style='text-align:left'>{html.escape(r['detail'] or '')}</td></tr>"
        for r in ev_rows)
    metric_rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{fmt(v, 4) if isinstance(v, float) else html.escape(str(v))}</td></tr>'
                          for k, v in metrics.items())
    plan_html = ''
    if run.plan:
        pv = run.plan.get('validation') or {}
        params = run.plan.get('params') or {}
        plan_html = (
            '<section><h2>Plan</h2>'
            f"<p>{len(run.plan.get('waypoints', []))} waypoints, predicted duration "
            f"{fmt((run.plan.get('prediction') or {}).get('duration'), 2, ' s')}, PMM generation "
            f"{fmt((run.plan.get('prediction') or {}).get('planning_ms'), 1, ' ms')}; "
            f"validation {'OK' if pv.get('ok') else 'FAILED'} (min clearance "
            f"{fmt(pv.get('min_clearance'))} m, min wall margin {fmt(pv.get('min_wall_margin'))} m)</p>"
            f"<p><small>{html.escape(json.dumps(params))}</small></p>"
            + ''.join(f"<p class='bad'>{html.escape(v)}</p>" for v in pv.get('violations', []))
            + '</section>')
    img = lambda key, title='': (f'<section><h2>{title}</h2><img src="data:image/png;base64,{figs[key]}"/></section>'  # noqa: E731
                                 if figs.get(key) else '')
    payload = json.dumps(replay_payload(run), separators=(',', ':'))
    meta = run.meta
    subtitle = (f"run <b>{html.escape(run.name)}</b> · course {html.escape(run.course.name)} · "
                f"{run.course.laps} laps × {run.course.gates_per_lap} gates"
                + (f" · commit {html.escape(meta.get('git_commit', ''))}" if meta.get('git_commit') else ''))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Race report {html.escape(run.name)}</title>
<style>{CSS}</style></head><body>
<header><h1>ICUAS 2027 · autonomous drone race report</h1><p>{subtitle}</p></header><main>
<div class="cards">{cards_html}</div>{reason}
<section class="replay"><h2>Replay</h2>
<div class="controls"><button id="rp-play">▶ play</button>
<select id="rp-rate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
<input type="range" id="rp-slider" min="0" max="0" value="0"></div>
<div id="rp-hud"></div>
<div class="grid2"><canvas id="rp-top" width="560" height="560"></canvas><canvas id="rp-side" width="560" height="220"></canvas></div>
<small>Top view (world x right, y up); the expected gate is highlighted; red squares = incidents.</small>
</section>
{img('topdown', 'Flown path')}
{img('profiles', 'Speed, height, acceleration')}
<section><h2>Time breakdown</h2>{table(phase_table(run), phase_cols)}<small>Race-manager phases after the start command (the race clock stops at touchdown).</small></section>
<section><h2>Laps</h2>{table(lap_table(run), lap_cols)}</section>
{img('laps', 'Lap and split times')}
<section><h2>Gates</h2>{table(gate_table(run), gate_cols)}</section>
{img('gate_offsets', 'Gate crossing accuracy')}
{img('clearance', 'Obstacle clearance')}
{img('tracking', 'Trajectory tracking')}
{img('3d', '3D view')}
{img('attitude', 'Attitude')}
{img('actuation', 'Actuation and controller timing')}
{img('speed_hist', 'Speed distribution')}
{plan_html}
<section><h2>Events</h2><table><thead>{ev_head}</thead><tbody>{ev_body}</tbody></table></section>
<section><h2>All metrics</h2><table><tbody>{metric_rows}</tbody></table></section>
</main><script>const REPLAY_DATA = {payload};{REPLAY_JS}</script></body></html>"""


def make_report(run_dir: str, out_dir: Optional[str] = None) -> str:
    run = load_run(run_dir)
    out_dir = out_dir or os.path.join(run.run_dir, 'report')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    metrics = compute_metrics(run)
    hull = float((run.summary.get('referee_params') or {}).get('hull_radius', 0.07))
    figs = {}
    if len(run.traj):
        figs['topdown'] = fig_topdown(run, fig_dir)
        figs['profiles'] = fig_profiles(run, fig_dir)
        figs['3d'] = fig_3d(run, fig_dir)
        figs['attitude'] = fig_attitude(run, fig_dir)
        figs['speed_hist'] = fig_speed_hist(run, fig_dir)
        figs['clearance'] = fig_clearance(run, fig_dir)
    figs['laps'] = fig_laps(run, fig_dir)
    figs['gate_offsets'] = fig_gate_offsets(run, fig_dir, hull)
    figs['tracking'] = fig_tracking(run, fig_dir)
    figs['actuation'] = fig_actuation(run, fig_dir)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, default=float)
    path = os.path.join(out_dir, 'report.html')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(build_html(run, metrics, figs))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Build the HTML analytics report of a race run.')
    ap.add_argument('run_dir', help='run directory (contains referee/summary.json)')
    ap.add_argument('--out', help='output directory (default: <run_dir>/report)')
    args = ap.parse_args(argv)
    path = make_report(args.run_dir, args.out)
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
