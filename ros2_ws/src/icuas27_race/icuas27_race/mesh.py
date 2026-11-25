"""Triangle-mesh loading (binary/ASCII STL and Wavefront OBJ) and obstacle distance queries."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import closest_points_on_triangles


def load_stl(path: str) -> np.ndarray:
    """Read an STL file (binary or ASCII). Returns triangles (N, 3, 3) float64."""
    with open(path, 'rb') as fh:
        data = fh.read()
    if len(data) >= 84:
        count = struct.unpack('<I', data[80:84])[0]
        if 84 + 50 * count == len(data):
            record = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                               ('attr', '<u2')])
            tris = np.frombuffer(data, dtype=record, count=count, offset=84)
            return tris['vertices'].astype(np.float64)
    # ASCII STL
    vertices: List[List[float]] = []
    for line in data.decode('ascii', errors='ignore').splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == 'vertex':
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError(f'cannot parse STL file {path}')
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read a Wavefront OBJ file.

    Returns (triangles (N, 3, 3), object index per triangle (N,), object names).
    Polygons are fan-triangulated. Vertex indices may be negative (relative).
    """
    vertices: List[Tuple[float, float, float]] = []
    triangles: List[Tuple[int, int, int]] = []
    owners: List[int] = []
    names: List[str] = []
    current = -1
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] in ('o', 'g'):
                names.append(' '.join(parts[1:]) if len(parts) > 1 else f'object{len(names)}')
                current = len(names) - 1
            elif parts[0] == 'f':
                if current < 0:
                    names.append('object0')
                    current = 0
                idx = []
                for token in parts[1:]:
                    i = int(token.split('/')[0])
                    idx.append(i - 1 if i > 0 else len(vertices) + i)
                for k in range(1, len(idx) - 1):
                    triangles.append((idx[0], idx[k], idx[k + 1]))
                    owners.append(current)
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    return v[t], np.asarray(owners, dtype=np.int64), names


def load_mesh(path: str) -> np.ndarray:
    """Load an STL or OBJ mesh as triangles (N, 3, 3)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.stl':
        return load_stl(path)
    if ext == '.obj':
        return load_obj(path)[0]
    raise ValueError(f'unsupported mesh format: {path}')


def remove_degenerate(triangles: np.ndarray, min_area: float = 1e-12) -> np.ndarray:
    """Drop zero-area triangles (they break the closest-point formulas)."""
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return triangles[area > min_area]


@dataclass
class ObstacleCluster:
    """A group of triangles (one gate structure) with a bounding sphere for culling."""

    name: str
    triangles: np.ndarray            # (N, 3, 3)
    center: np.ndarray = field(init=False)
    radius: float = field(init=False)
    lo: np.ndarray = field(init=False)
    hi: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        pts = self.triangles.reshape(-1, 3)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.lo, self.hi = lo, hi
        self.center = 0.5 * (lo + hi)
        self.radius = float(np.linalg.norm(pts - self.center, axis=1).max())


class ObstacleField:
    """Distance from points to a set of static obstacle meshes (the gates)."""

    def __init__(self, clusters: Sequence[ObstacleCluster]):
        self.clusters = [c for c in clusters if len(c.triangles) > 0]

    @classmethod
    def from_triangles(cls, triangles: np.ndarray, centers: Optional[Dict[str, np.ndarray]] = None
                       ) -> 'ObstacleField':
        """Group triangles by the nearest named centre (e.g. gate centres); one cluster if None."""
        tris = remove_degenerate(np.asarray(triangles, dtype=np.float64))
        if not centers:
            return cls([ObstacleCluster('obstacles', tris)])
        names = list(centers.keys())
        cen = np.stack([np.asarray(centers[n], dtype=float) for n in names])
        tri_c = tris.mean(axis=1)
        # Horizontal distance: gate legs reach the floor far below the opening centre.
        d = np.linalg.norm(tri_c[:, None, :2] - cen[None, :, :2], axis=2)
        owner = np.argmin(d, axis=1)
        return cls([ObstacleCluster(names[i], tris[owner == i]) for i in range(len(names))])

    def closest(self, point: np.ndarray, max_range: float = 0.6
                ) -> Tuple[float, Optional[np.ndarray], Optional[str]]:
        """Distance to the closest obstacle surface within max_range of the bounding spheres.

        Returns (distance, closest point, cluster name); (inf, None, None) if nothing is near.
        """
        p = np.asarray(point, dtype=float)
        best = (float('inf'), None, None)
        for cluster in self.clusters:
            if np.linalg.norm(p - cluster.center) > cluster.radius + max_range:
                continue
            tris = cluster.triangles
            cp, dist = closest_points_on_triangles(p, tris[:, 0], tris[:, 1], tris[:, 2])
            i = int(np.argmin(dist))
            if dist[i] < best[0]:
                best = (float(dist[i]), cp[i], cluster.name)
        return best

    def min_distance_batch(self, points: np.ndarray, max_range: float = 0.6) -> np.ndarray:
        """Distance of each point (N, 3) to the closest obstacle (inf when farther than range)."""
        pts = np.asarray(points, dtype=float)
        out = np.full(len(pts), np.inf)
        for cluster in self.clusters:
            # Distance to the cluster's bounding box is a lower bound of the distance to its
            # triangles (tighter than the bounding sphere for thin gate frames).
            gap = np.maximum(np.maximum(cluster.lo - pts, pts - cluster.hi), 0.0)
            near = np.flatnonzero(np.einsum('ij,ij->i', gap, gap) <= max_range * max_range)
            tris = cluster.triangles
            # Few large numpy calls: each call may hand the GIL to other threads (ROS callbacks).
            for k in range(0, len(near), 1024):
                idx = near[k:k + 1024]
                _, dist = closest_points_on_triangles(pts[idx][:, None, :], tris[:, 0],
                                                      tris[:, 1], tris[:, 2])
                out[idx] = np.minimum(out[idx], dist.min(axis=1))
        return out
