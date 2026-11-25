"""RViz marker builders shared by the race manager and the visualizer."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from .course import Course, Gate
from .geometry import quat_from_yaw

Color = Tuple[float, float, float, float]

GRAY: Color = (0.55, 0.55, 0.58, 1.0)
GREEN: Color = (0.15, 0.80, 0.30, 1.0)
YELLOW: Color = (1.00, 0.80, 0.10, 1.0)
RED: Color = (0.90, 0.15, 0.15, 1.0)
BLUE: Color = (0.20, 0.45, 0.95, 1.0)
WHITE: Color = (1.0, 1.0, 1.0, 1.0)


def rgba(c: Color) -> ColorRGBA:
    return ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(c[3]))


def turbo(value: float) -> Color:
    """Approximate 'turbo' colormap for value in [0, 1] (blue = slow, red = fast)."""
    x = float(np.clip(value, 0.0, 1.0))
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)), 1.0)


def _point(p: Sequence[float]) -> Point:
    return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))


def _quat(q: Sequence[float]) -> Quaternion:
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def base_marker(ns: str, mid: int, mtype: int, frame: str = 'world', stamp=None) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.lifetime = Duration(sec=0, nanosec=0)
    return m


def delete_all(frame: str = 'world') -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.action = Marker.DELETEALL
    return m


def speed_line(ns: str, mid: int, points: np.ndarray, speeds: np.ndarray, vmax: float,
               width: float = 0.03, frame: str = 'world') -> Marker:
    """LINE_STRIP coloured by speed."""
    m = base_marker(ns, mid, Marker.LINE_STRIP, frame)
    m.scale.x = width
    vmax = max(vmax, 1e-3)
    for p, v in zip(points, speeds):
        m.points.append(_point(p))
        m.colors.append(rgba(turbo(float(v) / vmax)))
    m.color = rgba(WHITE)
    return m


def gate_markers(gate: Gate, color: Color, mid: int, ns: str = 'gates',
                 frame: str = 'world') -> List[Marker]:
    """Four bars of the gate frame (CUBE markers) plus a travel-direction arrow."""
    out = []
    q = quat_from_yaw(gate.yaw)
    # Slightly larger than the real frame so the state colour covers the obstacle mesh.
    pad = 0.006
    bar = 0.5 * (gate.outer_width - gate.inner_width)
    half_w = 0.25 * (gate.outer_width + gate.inner_width)
    half_h = 0.25 * (gate.outer_height + gate.inner_height)
    depth = gate.depth + 2 * pad
    bars = [
        ((0.0, 0.0, half_h), (depth, gate.outer_width + 2 * pad, bar + 2 * pad)),
        ((0.0, 0.0, -half_h), (depth, gate.outer_width + 2 * pad, bar + 2 * pad)),
        ((0.0, half_w, 0.0), (depth, bar + 2 * pad, gate.inner_height)),
        ((0.0, -half_w, 0.0), (depth, bar + 2 * pad, gate.inner_height)),
    ]
    for k, (offset, size) in enumerate(bars):
        m = base_marker(ns, mid * 10 + k, Marker.CUBE, frame)
        m.pose.position = _point(gate.to_world(np.array(offset)))
        m.pose.orientation = _quat(q)
        m.scale = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
        m.color = rgba(color)
        out.append(m)
    arrow = base_marker(ns, mid * 10 + 5, Marker.ARROW, frame)
    tail = gate.center - 0.35 * gate.normal
    head = gate.center + 0.35 * gate.normal
    arrow.points = [_point(tail), _point(head)]
    arrow.scale = Vector3(x=0.025, y=0.06, z=0.08)
    arrow.color = rgba((color[0], color[1], color[2], 0.6))
    out.append(arrow)
    label = base_marker(ns, mid * 10 + 6, Marker.TEXT_VIEW_FACING, frame)
    label.pose.position = _point(gate.center + np.array([0.0, 0.0, 0.5 * gate.outer_height + 0.12]))
    label.scale.z = 0.14
    label.color = rgba(WHITE)
    label.text = f'G{gate.gate_id}'
    out.append(label)
    return out


def arena_markers(course: Course, frame: str = 'world') -> List[Marker]:
    lo, hi = course.arena_min, course.arena_max
    box = base_marker('arena', 0, Marker.LINE_LIST, frame)
    box.scale.x = 0.02
    box.color = rgba((0.8, 0.8, 0.8, 0.6))
    corners = [np.array([x, y, z]) for z in (lo[2], hi[2]) for y in (lo[1], hi[1])
               for x in (lo[0], hi[0])]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        box.points += [_point(corners[a]), _point(corners[b])]
    floor = base_marker('arena', 1, Marker.CUBE, frame)
    floor.pose.position = _point([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - 0.005])
    floor.scale = Vector3(x=float(hi[0] - lo[0]), y=float(hi[1] - lo[1]), z=0.01)
    floor.color = rgba((0.35, 0.40, 0.35, 0.5))
    pad = base_marker('arena', 2, Marker.CUBE, frame)
    pad.pose.position = _point(course.pad_center - np.array([0.0, 0.0, 0.5 * course.pad_center[2]]))
    pad.scale = Vector3(x=float(course.pad_size[0]), y=float(course.pad_size[1]),
                        z=float(max(course.pad_center[2], 0.01)))
    pad.color = rgba((0.75, 0.55, 0.30, 0.9))
    return [box, floor, pad]


def mesh_marker(ns: str, mid: int, resource: str, color: Optional[Color] = None,
                frame: str = 'world') -> Marker:
    m = base_marker(ns, mid, Marker.MESH_RESOURCE, frame)
    m.mesh_resource = resource
    m.scale = Vector3(x=1.0, y=1.0, z=1.0)
    if color is None:
        m.mesh_use_embedded_materials = True
        m.color = rgba((0.0, 0.0, 0.0, 0.0))
    else:
        m.color = rgba(color)
    return m


def sphere_list(ns: str, mid: int, points: Iterable[Sequence[float]], colors: Iterable[Color],
                size: float = 0.06, frame: str = 'world') -> Marker:
    m = base_marker(ns, mid, Marker.SPHERE_LIST, frame)
    m.scale = Vector3(x=size, y=size, z=size)
    for p, c in zip(points, colors):
        m.points.append(_point(p))
        m.colors.append(rgba(c))
    m.color = rgba(WHITE)
    return m


def text_marker(ns: str, mid: int, position: Sequence[float], text: str, size: float = 0.18,
                color: Color = WHITE, frame: str = 'world') -> Marker:
    m = base_marker(ns, mid, Marker.TEXT_VIEW_FACING, frame)
    m.pose.position = _point(position)
    m.scale.z = size
    m.color = rgba(color)
    m.text = text
    return m
