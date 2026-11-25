"""Loading of a race run directory (referee, plan and telemetry outputs) for analysis."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..course import Course
from ..geometry import quat_xyzw_to_matrices


def read_csv(path: str) -> Dict[str, np.ndarray]:
    """CSV with a header -> dict of float arrays (non-numeric columns kept as object arrays)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return {}
    header, body = rows[0], rows[1:]
    out: Dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        col = [r[j] if j < len(r) else '' for r in body]
        try:
            out[name] = np.array([float(x) if x != '' else math.nan for x in col])
        except ValueError:
            out[name] = np.array(col, dtype=object)
    return out


@dataclass
class RunData:
    run_dir: str
    summary: dict
    course: Course
    traj: Dict[str, np.ndarray]
    plan: Optional[dict] = None
    missions: List[dict] = field(default_factory=list)
    control: Dict[str, np.ndarray] = field(default_factory=dict)
    estimation: Dict[str, np.ndarray] = field(default_factory=dict)
    motors: Dict[str, np.ndarray] = field(default_factory=dict)
    planning: Dict[str, np.ndarray] = field(default_factory=dict)
    race_log: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.run_dir))

    @property
    def start(self) -> float:
        return float(self.summary.get('start_stamp') or (self.traj['stamp'][0] if self.traj else 0.0))

    def race_time(self, stamps: np.ndarray) -> np.ndarray:
        return np.asarray(stamps, dtype=float) - self.start

    # -- trajectory helpers ---------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return self.race_time(self.traj['stamp'])

    @property
    def p(self) -> np.ndarray:
        return np.stack([self.traj['x'], self.traj['y'], self.traj['z']], axis=1)

    @property
    def v(self) -> np.ndarray:
        return np.stack([self.traj['vx'], self.traj['vy'], self.traj['vz']], axis=1)

    @property
    def rotations(self) -> np.ndarray:
        q = np.stack([self.traj['qx'], self.traj['qy'], self.traj['qz'], self.traj['qw']], axis=1)
        return quat_xyzw_to_matrices(q)

    def alignment(self) -> Optional[np.ndarray]:
        if self.plan and self.plan.get('alignment'):
            return np.asarray(self.plan['alignment']['T_world_odom'], dtype=float)
        return None

    def mission_start(self) -> Optional[float]:
        """Race time at which the (first) race mission was dispatched."""
        for m in self.missions:
            if m.get('tag') == 'race':
                return float(m['stamp']) - self.start
        return None


def load_run(run_dir: str) -> RunData:
    run_dir = os.path.abspath(run_dir)
    summary_path = os.path.join(run_dir, 'referee', 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'{summary_path} not found: is this a race run directory?')
    with open(summary_path, encoding='utf-8') as fh:
        summary = json.load(fh)
    course = Course.from_dict(summary['course'])
    traj = read_csv(os.path.join(run_dir, 'referee', 'trajectory.csv'))
    plan = None
    for name in ('plan_preflight.json', 'plan_race.json'):
        path = os.path.join(run_dir, 'plan', name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as fh:
                plan = json.load(fh)
            break
    missions = []
    mpath = os.path.join(run_dir, 'plan', 'missions.csv')
    if os.path.exists(mpath):
        with open(mpath, newline='', encoding='utf-8') as fh:
            missions = list(csv.DictReader(fh))
    meta = {}
    mpath = os.path.join(run_dir, 'run.json')
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as fh:
            meta = json.load(fh)
    tel = os.path.join(run_dir, 'telemetry')
    return RunData(
        run_dir=run_dir, summary=summary, course=course, traj=traj, plan=plan,
        missions=missions,
        control=read_csv(os.path.join(tel, 'control.csv')),
        estimation=read_csv(os.path.join(tel, 'estimation.csv')),
        motors=read_csv(os.path.join(tel, 'motors.csv')),
        planning=read_csv(os.path.join(tel, 'planning.csv')),
        race_log=read_csv(os.path.join(tel, 'race.csv')),
        meta=meta,
    )


def find_runs(paths: List[str]) -> List[str]:
    """Expand directories/globs into run directories (those with referee/summary.json)."""
    out = []
    for p in paths:
        for candidate in sorted(glob.glob(p)) or [p]:
            if os.path.exists(os.path.join(candidate, 'referee', 'summary.json')):
                out.append(candidate)
            else:
                out += sorted(os.path.dirname(os.path.dirname(s)) for s in
                              glob.glob(os.path.join(candidate, '*', 'referee', 'summary.json')))
    return out
