"""Race referee (pure Python): scores a run from ground-truth samples following the ICUAS 2027
rulebook (sections 3.2, 4 and 10).

* A gate counts when the drone crosses its plane through the opening, in the travel direction,
  in race order and without touching the structure.
* Entering through the exit is an invalid (wrong-way) crossing and does not count.
* Hits on obstacles that do not end the flight and missed entries (passing beside the expected
  gate, or crossing it while touching it) add a penalty of `penalty_per_incident` seconds each;
  a missed gate must be retried legally, otherwise the run is incomplete.
* Leaving the flight envelope, a crash (drone on the floor or upside down) or exceeding the
  time limit invalidates the run.
* The run ends when the drone lands on the landing base after the last gate of the last lap.
  T_adj = T_elapsed + penalty_per_incident * (hits + missed_entries).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .course import Course
from .geometry import hull_support
from .mesh import ObstacleField

WAITING, RUNNING, FINISHED, INVALID = 'WAITING', 'RUNNING', 'FINISHED', 'INVALID'
STATE_CODES = {WAITING: 0, RUNNING: 1, FINISHED: 2, INVALID: 3}
EVENT_CODES = {'PASS': 0, 'HIT': 1, 'MISSED': 2, 'WRONG_WAY': 3, 'OUT_OF_ORDER': 4,
               'OUT_OF_BOUNDS': 5, 'CRASH': 6, 'FINISH': 7, 'TIMEOUT': 8}


@dataclass
class RefereeParams:
    hull_radius: float = 0.07            # rotor-tip radius of the Crazyflie hull (m)
    hull_half_height: float = 0.025      # half thickness of the hull disk (m)
    penalty_per_incident: float = 5.0    # rulebook eq. (1), delta t_j (s)
    time_limit: float = 300.0            # s after the start command
    beside_radius: float = 1.0           # plane crossings closer than this to a gate centre count
                                         # as passing beside it (m)
    contact_window: float = 0.15         # contact within +/- this of a crossing spoils it (s)
    contact_release: float = 0.01        # clearance that ends a contact episode (m)
    contact_release_time: float = 0.10   # s of clearance needed to end a contact episode
    landed_speed: float = 0.10           # m/s
    landed_height: float = 0.06          # m above the resting height measured on the base
    landed_settle_time: float = 0.30     # s
    crash_tilt_deg: float = 75.0
    crash_tilt_time: float = 0.20        # s
    floor_height: float = 0.10           # base_link height that means "lying on the floor" (m)
    airborne_height: float = 0.15        # m above the resting height: the drone took off
    bounds_margin: float = 0.0           # m (positive shrinks the flight envelope)


@dataclass
class RefereeEvent:
    type: str
    stamp: float                  # absolute time (s)
    race_time: float              # s since the start command
    gate_id: int = -1
    gate_index: int = -1
    lap: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_y: float = float('nan')
    offset_z: float = float('nan')
    speed: float = 0.0
    clearance: float = float('nan')
    detail: str = ''

    def to_dict(self) -> dict:
        d = asdict(self)
        d['position'] = list(self.position)
        return d


@dataclass
class _Contact:
    t_start: float
    t_last_negative: float
    min_clearance: float
    gate: str
    t_end: Optional[float] = None


@dataclass
class _PendingPass:
    index: int
    gate_id: int
    stamp: float
    position: np.ndarray
    offset_y: float
    offset_z: float
    speed: float
    deadline: float
    approach_angle: float = 0.0   # angle between the motion and the gate normal (deg)


class RefereeCore:
    def __init__(self, course: Course, params: Optional[RefereeParams] = None,
                 obstacles: Optional[ObstacleField] = None):
        self.course = course
        self.p = params or RefereeParams()
        self.obstacles = obstacles
        self.sequence = course.sequence
        self.state = WAITING
        self.reason = ''
        self.t_start: Optional[float] = None
        self.t_finish: Optional[float] = None
        self.next_index = 0
        self.hits = 0
        self.missed_entries = 0
        self.wrong_way = 0
        self.out_of_order = 0
        self.events: List[RefereeEvent] = []
        self.passes: List[dict] = []
        self.contacts: List[_Contact] = []
        self._contact: Optional[_Contact] = None
        self._pending: Optional[_PendingPass] = None
        self._needs_retry: Dict[int, int] = {}
        self._prev: Optional[Tuple[float, np.ndarray]] = None
        self._rest_z: Deque[float] = deque(maxlen=200)
        self.rest_height: Optional[float] = None
        self._airborne = False
        self._tilt_since: Optional[float] = None
        self._landed_since: Optional[float] = None
        self._clearance_hist: Deque[Tuple[float, float]] = deque(maxlen=400)
        self.min_clearance = math.inf
        self.distance = 0.0
        self.max_speed = 0.0
        self._speed_sum = 0.0
        self._speed_n = 0
        self.last_speed = 0.0
        self.last_clearance = math.inf

    # ------------------------------------------------------------------ public API
    @property
    def gates_total(self) -> int:
        return len(self.sequence)

    @property
    def penalty_time(self) -> float:
        return self.p.penalty_per_incident * (self.hits + self.missed_entries)

    def race_time(self, stamp: float) -> float:
        if self.t_start is None:
            return 0.0
        end = self.t_finish if self.t_finish is not None else stamp
        return max(0.0, end - self.t_start)

    def lap(self) -> int:
        return min(self.next_index // self.course.gates_per_lap + 1, self.course.laps)

    def start(self, stamp: float) -> None:
        if self.state != WAITING:
            return
        self.t_start = stamp
        self.state = RUNNING
        if self._rest_z:
            self.rest_height = float(np.median(np.asarray(self._rest_z)))

    def abort(self, stamp: float, reason: str) -> List[RefereeEvent]:
        if self.state in (FINISHED, INVALID):
            return []
        self._invalidate(stamp, reason)
        return []

    def update(self, stamp: float, position: np.ndarray, velocity: np.ndarray,
               rotation: np.ndarray) -> List[RefereeEvent]:
        """Feed one ground-truth sample (world frame). rotation = R_world_body (3, 3)."""
        p = np.asarray(position, dtype=float)
        v = np.asarray(velocity, dtype=float)
        speed = float(np.linalg.norm(v))
        self.last_speed = speed
        if self.state == WAITING:
            if speed < 0.05:
                self._rest_z.append(float(p[2]))
            self._prev = (stamp, p)
            return []
        if self.state != RUNNING:
            return []
        if self.rest_height is None:
            self.rest_height = float(np.median(np.asarray(self._rest_z))) if self._rest_z \
                else float(p[2])

        events: List[RefereeEvent] = []
        self._track_motion(stamp, p, speed)
        clearance, cluster = self._clearance(p, rotation)
        self._clearance_hist.append((stamp, clearance))
        events += self._update_contact(stamp, p, speed, clearance, cluster)
        events += self._resolve_pending(stamp)
        if self._prev is not None:
            events += self._crossings(self._prev[0], self._prev[1], stamp, p, speed)
        events += self._safety(stamp, p, speed, rotation)
        if self.state == RUNNING:
            events += self._finish(stamp, p, speed)
        if self.state == RUNNING and stamp - self.t_start > self.p.time_limit:
            events.append(self._event('TIMEOUT', stamp, p, speed,
                                      detail=f'no finish within {self.p.time_limit:.0f} s'))
            self._invalidate(stamp, 'time limit exceeded')
        self._prev = (stamp, p)
        self.events += events
        return events

    # ------------------------------------------------------------------ internals
    def _event(self, kind: str, stamp: float, p: np.ndarray, speed: float, **kw) -> RefereeEvent:
        return RefereeEvent(type=kind, stamp=stamp, race_time=self.race_time(stamp),
                            position=(float(p[0]), float(p[1]), float(p[2])), speed=speed, **kw)

    def _invalidate(self, stamp: float, reason: str) -> None:
        self.state = INVALID
        self.reason = reason
        self.t_finish = stamp

    def _track_motion(self, stamp: float, p: np.ndarray, speed: float) -> None:
        if self._prev is not None:
            self.distance += float(np.linalg.norm(p - self._prev[1]))
        self.max_speed = max(self.max_speed, speed)
        self._speed_sum += speed
        self._speed_n += 1
        if self.rest_height is not None and p[2] > self.rest_height + self.p.airborne_height:
            self._airborne = True

    def _clearance(self, p: np.ndarray, rotation: np.ndarray) -> Tuple[float, str]:
        if self.obstacles is None:
            return math.inf, ''
        dist, closest, name = self.obstacles.closest(p, max_range=0.4)
        if closest is None:
            return math.inf, ''
        if dist < 1e-9:
            return -self.p.hull_radius, name or ''
        u = (p - closest) / dist
        support = hull_support(rotation[:, 2], u, self.p.hull_radius, self.p.hull_half_height)
        clearance = dist - support
        self.last_clearance = clearance
        self.min_clearance = min(self.min_clearance, clearance)
        return clearance, name or ''

    def _gate_id_of(self, cluster: str) -> int:
        for gid, gate in self.course.gates.items():
            if gate.name == cluster or str(gid) == cluster:
                return gid
        return -1

    def _update_contact(self, stamp: float, p: np.ndarray, speed: float, clearance: float,
                        cluster: str) -> List[RefereeEvent]:
        events = []
        if clearance < 0.0:
            if self._contact is None:
                self._contact = _Contact(stamp, stamp, clearance, cluster)
                self.contacts.append(self._contact)
                self.hits += 1
                gid = self._gate_id_of(cluster)
                events.append(self._event('HIT', stamp, p, speed, gate_id=gid,
                                          clearance=clearance, lap=self.lap(),
                                          detail=f'contact with {cluster or "obstacle"}'))
            else:
                self._contact.t_last_negative = stamp
                self._contact.min_clearance = min(self._contact.min_clearance, clearance)
        elif self._contact is not None and clearance > self.p.contact_release and \
                stamp - self._contact.t_last_negative >= self.p.contact_release_time:
            self._contact.t_end = stamp
            self._contact = None
        return events

    def _contact_overlaps(self, t0: float, t1: float) -> bool:
        for c in self.contacts:
            end = c.t_end if c.t_end is not None else math.inf
            if c.t_start <= t1 and end >= t0:
                return True
        return False

    def _window_clearance(self, t0: float, t1: float) -> float:
        vals = [c for t, c in self._clearance_hist if t0 <= t <= t1]
        return float(min(vals)) if vals else float('nan')

    def _resolve_pending(self, stamp: float, force: bool = False) -> List[RefereeEvent]:
        pend = self._pending
        if pend is None or (stamp < pend.deadline and not force):
            return []
        self._pending = None
        w = self.p.contact_window
        window_clear = self._window_clearance(pend.stamp - w, pend.stamp + w)
        lap = self.course.lap_of(pend.index)
        common = dict(gate_id=pend.gate_id, gate_index=pend.index, lap=lap,
                      offset_y=pend.offset_y, offset_z=pend.offset_z, clearance=window_clear)
        if self._contact_overlaps(pend.stamp - w, pend.stamp + w):
            self.missed_entries += 1
            self._needs_retry[pend.index] = self._needs_retry.get(pend.index, 0) + 1
            ev = RefereeEvent(type='MISSED', stamp=pend.stamp, race_time=self.race_time(pend.stamp),
                              position=tuple(float(x) for x in pend.position), speed=pend.speed,
                              detail='crossed while touching the gate: entry does not count',
                              **common)
            return [ev]
        retry = self._needs_retry.pop(pend.index, 0)
        self.passes.append({
            'index': pend.index, 'gate_id': pend.gate_id, 'lap': lap,
            'stamp': pend.stamp, 'race_time': self.race_time(pend.stamp),
            'offset_y': pend.offset_y, 'offset_z': pend.offset_z, 'speed': pend.speed,
            'clearance': window_clear, 'retry': bool(retry),
            'approach_angle': pend.approach_angle,
            'position': [float(x) for x in pend.position],
        })
        self.next_index = max(self.next_index, pend.index + 1)
        ev = RefereeEvent(type='PASS', stamp=pend.stamp, race_time=self.race_time(pend.stamp),
                          position=tuple(float(x) for x in pend.position), speed=pend.speed,
                          detail='retry after missed entry' if retry else '', **common)
        return [ev]

    def _crossings(self, t0: float, p0: np.ndarray, t1: float, p1: np.ndarray,
                   speed: float) -> List[RefereeEvent]:
        events: List[RefereeEvent] = []
        expected = self.sequence[self.next_index] if self.next_index < self.gates_total else None
        for gid, gate in self.course.gates.items():
            s0 = float(np.dot(p0 - gate.center, gate.normal))
            s1 = float(np.dot(p1 - gate.center, gate.normal))
            forward = s0 < 0.0 <= s1
            backward = s0 >= 0.0 > s1
            if not (forward or backward):
                continue
            alpha = s0 / (s0 - s1)
            pc = p0 + alpha * (p1 - p0)
            tc = t0 + alpha * (t1 - t0)
            local = gate.to_gate(pc)
            y, z = float(local[1]), float(local[2])
            in_opening = abs(y) < 0.5 * gate.inner_width and abs(z) < 0.5 * gate.inner_height
            in_frame = abs(y) < 0.5 * gate.outer_width and abs(z) < 0.5 * gate.outer_height
            near = math.hypot(y, z) < self.p.beside_radius
            if backward:
                if in_frame:
                    self.wrong_way += 1
                    events.append(self._event('WRONG_WAY', tc, pc, speed, gate_id=gid,
                                              lap=self.lap(), offset_y=y, offset_z=z,
                                              detail='entered through the exit'))
                continue
            if gid == expected:
                if in_opening:
                    if self._pending is not None:
                        events += self._resolve_pending(tc, force=True)
                    step = p1 - p0
                    norm = float(np.linalg.norm(step))
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(
                        np.dot(step, gate.normal)) / norm)))) if norm > 1e-9 else 0.0
                    self._pending = _PendingPass(self.next_index, gid, tc, pc, y, z, speed,
                                                 tc + self.p.contact_window, angle)
                elif near:
                    self.missed_entries += 1
                    self._needs_retry[self.next_index] = self._needs_retry.get(self.next_index, 0) + 1
                    why = 'through the frame' if in_frame else 'beside the gate'
                    events.append(self._event('MISSED', tc, pc, speed, gate_id=gid,
                                              gate_index=self.next_index, lap=self.lap(),
                                              offset_y=y, offset_z=z,
                                              detail=f'crossed the gate plane {why}'))
            elif in_opening:
                self.out_of_order += 1
                events.append(self._event('OUT_OF_ORDER', tc, pc, speed, gate_id=gid,
                                          lap=self.lap(), offset_y=y, offset_z=z,
                                          detail=f'expected gate {expected}'))
        return events

    def _safety(self, stamp: float, p: np.ndarray, speed: float,
                rotation: np.ndarray) -> List[RefereeEvent]:
        if not self.course.inside_arena(p, self.p.bounds_margin):
            self._invalidate(stamp, 'left the flight envelope')
            return [self._event('OUT_OF_BOUNDS', stamp, p, speed, lap=self.lap(),
                                detail='left the flight envelope')]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
        if tilt > self.p.crash_tilt_deg:
            if self._tilt_since is None:
                self._tilt_since = stamp
            elif stamp - self._tilt_since >= self.p.crash_tilt_time:
                self._invalidate(stamp, f'tilt {tilt:.0f} deg')
                return [self._event('CRASH', stamp, p, speed, lap=self.lap(),
                                    detail=f'attitude lost (tilt {tilt:.0f} deg)')]
        else:
            self._tilt_since = None
        if self._airborne and p[2] < self.p.floor_height and not self.course.over_pad(p):
            self._invalidate(stamp, 'on the floor')
            return [self._event('CRASH', stamp, p, speed, lap=self.lap(),
                                detail='drone on the floor outside the landing base')]
        return []

    def _finish(self, stamp: float, p: np.ndarray, speed: float) -> List[RefereeEvent]:
        if self.next_index < self.gates_total or self._pending is not None:
            self._landed_since = None
            return []
        rest = self.rest_height if self.rest_height is not None else 0.0
        landed = (self.course.over_pad(p) and p[2] <= rest + self.p.landed_height and
                  speed < self.p.landed_speed)
        if not landed:
            self._landed_since = None
            return []
        if self._landed_since is None:
            self._landed_since = stamp
            return []
        if stamp - self._landed_since < self.p.landed_settle_time:
            return []
        self.t_finish = self._landed_since
        self.state = FINISHED
        return [self._event('FINISH', self._landed_since, p, speed, lap=self.course.laps,
                            detail='landed on the landing base')]

    # ------------------------------------------------------------------ results
    def lap_times(self) -> Dict[str, List[float]]:
        """Lap times from the passes.

        * `by_last_gate`: lap i ends when its last gate is passed (lap 1 includes takeoff).
        * `flying`: time between consecutive passes of the first gate of each lap.
        """
        gpl = self.course.gates_per_lap
        by_index = {p['index']: p['stamp'] for p in self.passes}
        last, flying = [], []
        prev = self.t_start
        for lap in range(self.course.laps):
            end_idx = (lap + 1) * gpl - 1
            if end_idx in by_index and prev is not None:
                last.append(by_index[end_idx] - prev)
                prev = by_index[end_idx]
            else:
                break
        for lap in range(self.course.laps - 1):
            a, b = lap * gpl, (lap + 1) * gpl
            if a in by_index and b in by_index:
                flying.append(by_index[b] - by_index[a])
        return {'by_last_gate': last, 'flying': flying}

    def summary(self) -> dict:
        elapsed = self.race_time(self.t_finish if self.t_finish is not None else
                                 (self._prev[0] if self._prev else 0.0))
        valid = self.state == FINISHED
        laps_done = self.next_index // self.course.gates_per_lap
        return {
            'state': self.state,
            'valid': valid,
            'reason': self.reason,
            'start_stamp': self.t_start,
            'finish_stamp': self.t_finish,
            'elapsed_time': elapsed,
            'hits': self.hits,
            'missed_entries': self.missed_entries,
            'wrong_way': self.wrong_way,
            'out_of_order': self.out_of_order,
            'penalty_per_incident': self.p.penalty_per_incident,
            'penalty_time': self.penalty_time,
            'adjusted_time': elapsed + self.penalty_time if valid else None,
            'gates_passed': len(self.passes),
            'gates_total': self.gates_total,
            'laps_completed': laps_done,
            'laps_total': self.course.laps,
            'lap_times': self.lap_times(),
            'passes': list(self.passes),
            'contacts': [{'t_start': c.t_start, 't_end': c.t_end, 'min_clearance': c.min_clearance,
                          'obstacle': c.gate} for c in self.contacts],
            'min_clearance': None if math.isinf(self.min_clearance) else self.min_clearance,
            'distance_flown': self.distance,
            'max_speed': self.max_speed,
            'mean_speed': self._speed_sum / self._speed_n if self._speed_n else 0.0,
            'rest_height': self.rest_height,
            'events': [e.to_dict() for e in self.events],
        }
