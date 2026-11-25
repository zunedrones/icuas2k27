"""Race plan: gate waypoints for every lap, recovery manoeuvres, frame alignment between the
course (world) and the LUS estimation frame, and validation of a predicted trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .course import Course, Gate
from .geometry import (average_quaternions_xyzw, invert_transform, make_transform,
                       matrix_to_quat_xyzw, quat_xyzw_to_matrix, transform_points, wrap_angle,
                       yaw_of_matrix)
from .mesh import ObstacleField


@dataclass
class PlanParams:
    approach_distance: float = 0.30   # pre-gate waypoint distance before the gate plane (m)
    exit_distance: float = 0.30       # post-gate waypoint distance after the gate plane (m)
    heading_mode: str = 'constant'    # 'constant' (fixed world yaw) or 'gate' (face each gate)
    constant_heading: float = 0.0     # world yaw for 'constant' mode (rad)
    final_height: float = 0.30        # final hover height above the landing base (m)
    recovery_lateral: float = 0.75    # lateral offset when flying around a missed gate (m)
    recovery_depth: float = 0.60      # depth before/after the gate plane during a recovery (m)

    def __post_init__(self) -> None:
        if self.heading_mode not in ('constant', 'gate'):
            raise ValueError("heading_mode must be 'constant' or 'gate'")
        if self.approach_distance < 0.0 or self.exit_distance < 0.0:
            raise ValueError('approach/exit distances must be >= 0')


@dataclass
class Waypoint:
    position: np.ndarray       # world (3,)
    heading: float             # world yaw (rad)
    label: str
    sequence_index: int        # race sequence index of the related gate, -1 if none
    kind: str                  # pre | gate | post | final | recovery

    def to_dict(self) -> dict:
        return {'position': self.position.tolist(), 'heading': self.heading,
                'label': self.label, 'sequence_index': self.sequence_index, 'kind': self.kind}


def _heading(gate: Gate, params: PlanParams) -> float:
    return gate.yaw if params.heading_mode == 'gate' else params.constant_heading


def gate_waypoints(course: Course, sequence_index: int, params: PlanParams) -> List[Waypoint]:
    """Pre-gate, gate-centre and post-gate waypoints of one race sequence entry."""
    gate = course.gates[course.sequence[sequence_index]]
    lap = course.lap_of(sequence_index)
    tag = f'L{lap}G{gate.gate_id}'
    heading = _heading(gate, params)
    out = []
    if params.approach_distance > 0.0:
        out.append(Waypoint(gate.center - params.approach_distance * gate.normal, heading,
                            f'{tag}-pre', sequence_index, 'pre'))
    out.append(Waypoint(gate.center.copy(), heading, tag, sequence_index, 'gate'))
    if params.exit_distance > 0.0:
        out.append(Waypoint(gate.center + params.exit_distance * gate.normal, heading,
                            f'{tag}-post', sequence_index, 'post'))
    return out


def final_waypoint(course: Course, params: PlanParams) -> Waypoint:
    """Hover point above the landing base where the landing manoeuvre starts."""
    p = course.pad_center + np.array([0.0, 0.0, params.final_height])
    heading = params.constant_heading if params.heading_mode == 'constant' else \
        course.gates[course.sequence[-1]].yaw
    return Waypoint(p, heading, 'final', -1, 'final')


def race_waypoints(course: Course, params: PlanParams, start_index: int = 0,
                   end_index: Optional[int] = None) -> List[Waypoint]:
    """Waypoints of race sequence entries [start_index, end_index); the final hover point is
    appended when the slice reaches the end of the race (end_index None = whole race)."""
    n = len(course.sequence)
    end = n if end_index is None else min(max(end_index, start_index), n)
    if not 0 <= start_index <= n:
        raise ValueError('start_index outside the race sequence')
    out: List[Waypoint] = []
    for k in range(start_index, end):
        out.extend(gate_waypoints(course, k, params))
    if end == n:
        out.append(final_waypoint(course, params))
    return out


def recovery_waypoints(course: Course, params: PlanParams, sequence_index: int,
                       drone_world: np.ndarray, side: Optional[int] = None,
                       end_index: Optional[int] = None) -> List[Waypoint]:
    """Re-enter the missed gate `sequence_index` from its entry side, then finish the race.

    When the drone is already at or past the gate plane it flies around the frame (never
    through it backwards, which would be a wrong-way crossing): first to a point beside the
    exit side, then beside the entry side, then into the normal gate approach.
    `side` (+1 left of travel / -1 right) is chosen from the drone position when None.
    """
    gate = course.gates[course.sequence[sequence_index]]
    local = gate.to_gate(drone_world)
    heading = _heading(gate, params)
    out: List[Waypoint] = []
    if local[0] > -0.05:
        if side is None:
            side = 1 if local[1] >= 0.0 else -1
        lat = side * params.recovery_lateral
        exit_x = max(params.recovery_depth, min(local[0], 1.5))
        for x, name in ((exit_x, 'around-exit'), (-params.recovery_depth, 'around-entry')):
            out.append(Waypoint(gate.to_world(np.array([x, lat, 0.0])), heading,
                                f'R{sequence_index}-{name}', sequence_index, 'recovery'))
    out.extend(race_waypoints(course, params, sequence_index, end_index))
    return out


def trim_passed_waypoints(course: Course, waypoints: Sequence[Waypoint], position: np.ndarray,
                          margin: float = 0.05) -> List[Waypoint]:
    """Drop leading gate waypoints (pre, centre, post) that `position` has already passed along
    their gate normal. A fly-through mission must never start behind the reference: the planner
    would fly back to them, through the gate's exit."""
    out = list(waypoints)
    while out and out[0].kind in ('pre', 'gate', 'post') and out[0].sequence_index >= 0:
        gate = course.gates[course.sequence[out[0].sequence_index]]
        if gate.to_gate(position)[0] >= gate.to_gate(out[0].position)[0] - margin:
            out.pop(0)
        else:
            break
    return out


def drop_close_waypoints(waypoints: Sequence[Waypoint], start: np.ndarray,
                         min_distance: float = 0.05) -> List[Waypoint]:
    """Remove waypoints closer than min_distance to the previous one (PMM rejects coincident
    fly-through waypoints) and the leading ones that coincide with the start."""
    out: List[Waypoint] = []
    last = np.asarray(start, dtype=float)
    for wp in waypoints:
        if np.linalg.norm(wp.position - last) >= min_distance:
            out.append(wp)
            last = wp.position
    return out


class FrameAlignment:
    """Transform between the course frame (world) and the LUS estimation frame (odom).

    T_world_odom is the pose of the estimation frame in the world: p_world = T_world_odom p_odom.
    """

    def __init__(self, t_world_odom: Optional[np.ndarray] = None, source: str = 'identity'):
        self.t_world_odom = np.eye(4) if t_world_odom is None else np.asarray(t_world_odom, float)
        self.t_odom_world = invert_transform(self.t_world_odom)
        self.source = source

    @property
    def yaw(self) -> float:
        return yaw_of_matrix(self.t_world_odom[:3, :3])

    def point_to_odom(self, p_world: np.ndarray) -> np.ndarray:
        return transform_points(self.t_odom_world, p_world)

    def point_to_world(self, p_odom: np.ndarray) -> np.ndarray:
        return transform_points(self.t_world_odom, p_odom)

    def vector_to_world(self, v_odom: np.ndarray) -> np.ndarray:
        return np.asarray(v_odom, dtype=float) @ self.t_world_odom[:3, :3].T

    def heading_to_odom(self, yaw_world: float) -> float:
        return wrap_angle(yaw_world - self.yaw)

    def heading_to_world(self, yaw_odom: float) -> float:
        return wrap_angle(yaw_odom + self.yaw)

    @staticmethod
    def from_pose_pairs(world_body: Sequence[np.ndarray], odom_body: Sequence[np.ndarray],
                        source: str = 'ground_truth') -> 'FrameAlignment':
        """Average T_world_odom = T_world_body inv(T_odom_body) over synchronised samples."""
        if len(world_body) == 0 or len(world_body) != len(odom_body):
            raise ValueError('need matching non-empty pose lists')
        translations, quats = [], []
        for t_wb, t_ob in zip(world_body, odom_body):
            t_wo = t_wb @ invert_transform(t_ob)
            translations.append(t_wo[:3, 3])
            quats.append(matrix_to_quat_xyzw(t_wo[:3, :3]))
        r = quat_xyzw_to_matrix(average_quaternions_xyzw(np.asarray(quats)))
        return FrameAlignment(make_transform(r, np.mean(translations, axis=0)), source)

    def to_dict(self) -> dict:
        return {'source': self.source, 'T_world_odom': self.t_world_odom.tolist(),
                'yaw': self.yaw}


@dataclass
class ValidationParams:
    hull_radius: float = 0.07        # drone hull radius used for clearance (m)
    min_clearance: float = 0.06      # required hull-to-obstacle distance along the plan (m)
    wall_margin: float = 0.20        # required distance to the arena walls (m)
    ceiling_margin: float = 0.20
    min_height: float = 0.15         # above the floor away from the landing base (m)
    sample_stride: int = 2           # check every n-th sample of the prediction
    max_thrust_accel: float = 14.0   # |a - g| the airframe can produce (m/s^2), 0 = unchecked
    max_thrust_tilt_deg: float = 180.0  # angle of the thrust vector from vertical, 180 = unchecked


@dataclass
class ValidationReport:
    ok: bool
    min_clearance: float
    min_wall_margin: float
    min_height: float
    max_height: float
    max_speed: float
    duration: float
    violations: List[str] = field(default_factory=list)
    max_thrust_accel: float = float('nan')
    max_thrust_tilt_deg: float = float('nan')

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'min_clearance': self.min_clearance,
                'min_wall_margin': self.min_wall_margin, 'min_height': self.min_height,
                'max_height': self.max_height, 'max_speed': self.max_speed,
                'duration': self.duration, 'violations': list(self.violations),
                'max_thrust_accel': self.max_thrust_accel,
                'max_thrust_tilt_deg': self.max_thrust_tilt_deg}


def validate_trajectory(course: Course, obstacles: Optional[ObstacleField],
                        positions_world: np.ndarray, velocities_world: np.ndarray,
                        times: np.ndarray, params: ValidationParams,
                        accelerations_world: Optional[np.ndarray] = None) -> ValidationReport:
    """Check a predicted reference trajectory (world frame) against the arena and obstacles."""
    pos = np.asarray(positions_world, dtype=float)
    vel = np.asarray(velocities_world, dtype=float)
    idx = np.arange(0, len(pos), max(1, params.sample_stride))
    if len(pos) and idx[-1] != len(pos) - 1:
        idx = np.append(idx, len(pos) - 1)
    p = pos[idx]
    violations: List[str] = []

    lo = course.arena_min
    hi = course.arena_max
    wall = np.minimum(np.minimum(p[:, 0] - lo[0], hi[0] - p[:, 0]),
                      np.minimum(p[:, 1] - lo[1], hi[1] - p[:, 1]))
    min_wall = float(wall.min()) if len(wall) else math.inf
    if min_wall < params.wall_margin:
        k = int(np.argmin(wall))
        violations.append(f'wall margin {min_wall:.3f} m < {params.wall_margin:.3f} m at '
                          f't={times[idx[k]]:.2f}s p={np.round(p[k], 3).tolist()}')
    max_h = float(p[:, 2].max()) if len(p) else 0.0
    if max_h > hi[2] - params.ceiling_margin:
        violations.append(f'height {max_h:.3f} m above ceiling limit {hi[2] - params.ceiling_margin:.3f} m')
    half = 0.5 * course.pad_size + 0.3
    away = np.any(np.abs(p[:, :2] - np.asarray(course.pad_center)[:2]) > half[:2], axis=1)
    min_h = float(p[away, 2].min()) if np.any(away) else float(p[:, 2].min())
    if np.any(away) and min_h < params.min_height:
        violations.append(f'height {min_h:.3f} m below {params.min_height:.3f} m away from the base')

    min_clear = math.inf
    if obstacles is not None and len(p):
        dist = obstacles.min_distance_batch(p, max_range=params.min_clearance + params.hull_radius + 0.5)
        clear = dist - params.hull_radius
        min_clear = float(clear.min())
        if min_clear < params.min_clearance:
            k = int(np.argmin(clear))
            violations.append(f'obstacle clearance {min_clear:.3f} m < {params.min_clearance:.3f} m '
                              f'at t={times[idx[k]]:.2f}s p={np.round(p[k], 3).tolist()}')
    max_thrust = max_tilt = float('nan')
    if accelerations_world is not None and len(accelerations_world):
        thrust = np.asarray(accelerations_world, dtype=float) + np.array([0.0, 0.0, 9.81])
        norm = np.linalg.norm(thrust, axis=1)
        tilt = np.degrees(np.arccos(np.clip(thrust[:, 2] / np.maximum(norm, 1e-9), -1.0, 1.0)))
        max_thrust, max_tilt = float(norm.max()), float(tilt.max())
        if params.max_thrust_accel > 0 and max_thrust > params.max_thrust_accel:
            k = int(np.argmax(norm))
            violations.append(f'thrust acceleration {max_thrust:.2f} m/s^2 > '
                              f'{params.max_thrust_accel:.2f} at t={times[k]:.2f}s')
        if max_tilt > params.max_thrust_tilt_deg:
            k = int(np.argmax(tilt))
            dt = np.diff(times, append=times[-1])
            violations.append(f'thrust tilt {max_tilt:.0f} deg > {params.max_thrust_tilt_deg:.0f} '
                              f'at t={times[k]:.2f}s ({dt[tilt > params.max_thrust_tilt_deg].sum():.2f} s '
                              f'in total)')
    speed = np.linalg.norm(vel, axis=1) if len(vel) else np.zeros(1)
    return ValidationReport(
        ok=not violations,
        min_clearance=min_clear,
        min_wall_margin=min_wall,
        min_height=min_h,
        max_height=max_h,
        max_speed=float(speed.max()),
        duration=float(times[-1] - times[0]) if len(times) else 0.0,
        violations=violations,
        max_thrust_accel=max_thrust,
        max_thrust_tilt_deg=max_tilt,
    )
