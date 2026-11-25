"""Race course model: arena, landing base, gates (pose, size, travel direction) and gate order."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml

from .geometry import make_transform, rot_z


def resolve_path(uri: str, relative_to: Optional[str] = None) -> str:
    """Resolve ``package://pkg/rel``, ``file://`` or a (relative) filesystem path."""
    if uri.startswith('package://'):
        rest = uri[len('package://'):]
        pkg, _, rel = rest.partition('/')
        from ament_index_python.packages import get_package_share_directory  # lazy: ROS only
        return os.path.join(get_package_share_directory(pkg), rel)
    if uri.startswith('file://'):
        return uri[len('file://'):]
    path = os.path.expanduser(os.path.expandvars(uri))
    if not os.path.isabs(path) and relative_to:
        path = os.path.join(relative_to, path)
    return path


@dataclass
class Gate:
    """Rectangular gate. Gate frame: origin at the opening centre, +x = direction of travel
    (entry side x < 0, exit side x > 0), +y = left of travel, +z = up."""

    gate_id: int
    name: str
    center: np.ndarray            # (3,) world
    yaw: float                    # direction of travel, world frame (rad)
    inner_width: float
    inner_height: float
    outer_width: float
    outer_height: float
    depth: float

    @property
    def rotation(self) -> np.ndarray:
        """R_world_gate (3, 3)."""
        return rot_z(self.yaw)

    @property
    def transform(self) -> np.ndarray:
        """T_world_gate (4, 4)."""
        return make_transform(self.rotation, self.center)

    @property
    def normal(self) -> np.ndarray:
        """Unit direction of travel through the gate (world)."""
        return self.rotation[:, 0]

    def to_gate(self, points: np.ndarray) -> np.ndarray:
        """World points (N, 3) or (3,) -> gate frame: R^T (p - c)."""
        return (np.asarray(points, dtype=float) - self.center) @ self.rotation

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Gate-frame points -> world: R p + c."""
        return np.asarray(points, dtype=float) @ self.rotation.T + self.center

    def corners(self, inner: bool = True) -> np.ndarray:
        """Opening (inner) or frame (outer) corners in world, counter-clockwise from the
        bottom-right seen from the entry side. (4, 3)."""
        w = 0.5 * (self.inner_width if inner else self.outer_width)
        h = 0.5 * (self.inner_height if inner else self.outer_height)
        local = np.array([[0.0, -w, -h], [0.0, w, -h], [0.0, w, h], [0.0, -w, h]])
        return self.to_world(local)


@dataclass
class Course:
    name: str
    laps: int
    arena_min: np.ndarray
    arena_max: np.ndarray
    pad_center: np.ndarray        # top surface centre of the takeoff/landing base (world)
    pad_size: np.ndarray          # (width x, depth y)
    start_position: np.ndarray    # drone spawn position (world); also the ground-truth offset
    start_rpy: np.ndarray
    gates: Dict[int, Gate]
    order: List[int]
    obstacle_mesh: Optional[str] = None
    source_file: Optional[str] = None
    description: str = ''
    _sequence: List[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        missing = [g for g in self.order if g not in self.gates]
        if missing:
            raise ValueError(f'gate order references unknown gates {missing}')
        if self.laps < 1:
            raise ValueError('laps must be >= 1')
        if not np.all(self.arena_max > self.arena_min):
            raise ValueError('arena max must exceed arena min on every axis')
        self._sequence = [g for _ in range(self.laps) for g in self.order]

    @property
    def sequence(self) -> List[int]:
        """Gate ids in race order for all laps (laps x len(order))."""
        return list(self._sequence)

    @property
    def gates_per_lap(self) -> int:
        return len(self.order)

    def lap_of(self, sequence_index: int) -> int:
        """1-based lap of a 0-based race sequence index."""
        return sequence_index // self.gates_per_lap + 1

    def inside_arena(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.arena_min + margin) and np.all(p <= self.arena_max - margin))

    def over_pad(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p, dtype=float)
        half = 0.5 * self.pad_size - margin
        return bool(abs(p[0] - self.pad_center[0]) <= half[0] and
                    abs(p[1] - self.pad_center[1]) <= half[1])

    def obstacle_mesh_path(self) -> Optional[str]:
        if not self.obstacle_mesh:
            return None
        base = os.path.dirname(self.source_file) if self.source_file else None
        return resolve_path(self.obstacle_mesh, base)

    @staticmethod
    def load(path: str) -> 'Course':
        with open(path, 'r', encoding='utf-8') as fh:
            doc = yaml.safe_load(fh)
        course = Course.from_dict(doc['course'] if 'course' in doc else doc)
        course.source_file = os.path.abspath(path)
        return course

    @staticmethod
    def from_dict(d: dict) -> 'Course':
        defaults = d.get('gate_defaults', {})
        gates: Dict[int, Gate] = {}
        for g in d['gates']:
            spec = dict(defaults)
            spec.update(g)
            gate = Gate(
                gate_id=int(spec['id']),
                name=str(spec.get('name', f"G{spec['id']}")),
                center=np.asarray(spec['center'], dtype=float),
                yaw=float(spec['yaw']),
                inner_width=float(spec['inner_width']),
                inner_height=float(spec['inner_height']),
                outer_width=float(spec['outer_width']),
                outer_height=float(spec['outer_height']),
                depth=float(spec['depth']),
            )
            if gate.gate_id in gates:
                raise ValueError(f'duplicate gate id {gate.gate_id}')
            if not (0 < gate.inner_width < gate.outer_width and
                    0 < gate.inner_height < gate.outer_height):
                raise ValueError(f'gate {gate.gate_id}: inner size must be smaller than outer')
            gates[gate.gate_id] = gate
        start = d.get('start', {})
        return Course(
            name=str(d.get('name', 'course')),
            description=str(d.get('description', '')),
            laps=int(d.get('laps', 1)),
            arena_min=np.asarray(d['arena']['min'], dtype=float),
            arena_max=np.asarray(d['arena']['max'], dtype=float),
            pad_center=np.asarray(d['landing_pad']['center'], dtype=float),
            pad_size=np.asarray(d['landing_pad']['size'], dtype=float),
            start_position=np.asarray(start.get('position', [0.0, 0.0, 0.0]), dtype=float),
            start_rpy=np.asarray(start.get('rpy', [0.0, 0.0, 0.0]), dtype=float),
            gates=gates,
            order=[int(x) for x in d.get('order', sorted(gates))],
            obstacle_mesh=d.get('obstacle_mesh'),
        )

    def to_dict(self) -> dict:
        """Serialisable summary (used in plan.json / reports)."""
        return {
            'name': self.name,
            'laps': self.laps,
            'arena': {'min': self.arena_min.tolist(), 'max': self.arena_max.tolist()},
            'landing_pad': {'center': self.pad_center.tolist(), 'size': self.pad_size.tolist()},
            'start': {'position': self.start_position.tolist(), 'rpy': self.start_rpy.tolist()},
            'order': list(self.order),
            'gates': [{
                'id': g.gate_id, 'name': g.name, 'center': g.center.tolist(), 'yaw': g.yaw,
                'inner_width': g.inner_width, 'inner_height': g.inner_height,
                'outer_width': g.outer_width, 'outer_height': g.outer_height, 'depth': g.depth,
            } for g in self.gates.values()],
        }
