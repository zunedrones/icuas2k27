"""RViz visualisation of the race: course, drone, flown path coloured by speed, gate states,
referee events and a head-up display. Works live and while replaying a recorded bag."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from icuas27_msgs.msg import GateEvent, RaceEvent, RaceStatus, RefereeStatus
from icuas27_race import viz
from icuas27_race.course import Course
from icuas27_race.ros_utils import (EVENTS_QOS, RELIABLE_QOS, SENSOR_QOS, GroundTruthAdapter,
                                    expand)

EVENT_COLORS = {
    GateEvent.PASS: viz.GREEN, GateEvent.HIT: viz.RED, GateEvent.MISSED: (1.0, 0.45, 0.0, 1.0),
    GateEvent.WRONG_WAY: (0.8, 0.0, 0.8, 1.0), GateEvent.OUT_OF_ORDER: (0.6, 0.3, 1.0, 1.0),
    GateEvent.OUT_OF_BOUNDS: viz.RED, GateEvent.CRASH: viz.RED, GateEvent.FINISH: viz.WHITE,
    GateEvent.TIMEOUT: viz.RED,
}
REFEREE_STATES = {0: 'WAITING', 1: 'RUNNING', 2: 'FINISHED', 3: 'INVALID'}


class Visualizer(Node):
    def __init__(self) -> None:
        super().__init__('race_visualizer')
        self.declare_parameter('uav_name', 'uav1')
        self.declare_parameter('course_file', '')
        self.declare_parameter('ground_truth_topic', '/uav1/ground_truth')
        self.declare_parameter('drone_mesh',
                               'package://laser_uav_simulation/models/laser_uavs_description/'
                               'meshes/crazyflie/crazyflie.dae')
        self.declare_parameter('drone_scale', 3.0)
        self.declare_parameter('path_resolution', 0.02)
        self.declare_parameter('max_speed_color', 4.0)
        self.declare_parameter('rate', 20.0)

        self.uav = self.get_parameter('uav_name').value
        course_file = self.get_parameter('course_file').value
        if not course_file:
            raise RuntimeError('course_file parameter is required')
        self.course = Course.load(expand(course_file))
        self.adapter = GroundTruthAdapter(self.course.start_position, self.course.start_rpy)
        self.drone_mesh = self.get_parameter('drone_mesh').value
        self.drone_scale = float(self.get_parameter('drone_scale').value)
        self.res = float(self.get_parameter('path_resolution').value)
        self.vmax = float(self.get_parameter('max_speed_color').value)

        self.state = None
        self.path: List[np.ndarray] = []
        self.speeds: List[float] = []
        self.events: List[GateEvent] = []
        self.referee: Optional[RefereeStatus] = None
        self.race: Optional[RaceStatus] = None
        self.missed: Dict[int, bool] = {}
        self.dirty_static = True
        self.cycle = 0

        self.pub = self.create_publisher(MarkerArray, '/race/markers', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, self.get_parameter('ground_truth_topic').value,
                                 self.on_gt, SENSOR_QOS)
        self.create_subscription(RefereeStatus, '/race/referee/status',
                                 lambda m: setattr(self, 'referee', m), RELIABLE_QOS)
        self.create_subscription(RaceStatus, '/race/status',
                                 lambda m: setattr(self, 'race', m), RELIABLE_QOS)
        self.create_subscription(GateEvent, '/race/referee/events', self.on_event, EVENTS_QOS)
        self.create_subscription(RaceEvent, '/race/events', self.on_race_event, EVENTS_QOS)
        self.create_timer(1.0 / float(self.get_parameter('rate').value), self.publish)
        self.create_timer(2.0, self.mark_static)

    def mark_static(self) -> None:
        self.dirty_static = True

    def on_race_event(self, msg: RaceEvent) -> None:
        if msg.type == RaceEvent.START:
            self.path, self.speeds, self.events, self.missed = [], [], [], {}

    def on_event(self, msg: GateEvent) -> None:
        self.events.append(msg)
        if msg.type == GateEvent.MISSED:
            self.missed[msg.gate_id] = True
        elif msg.type == GateEvent.PASS:
            self.missed.pop(msg.gate_id, None)

    def on_gt(self, msg: Odometry) -> None:
        self.state = self.adapter.convert(msg)
        p = self.state.position
        if not self.path or np.linalg.norm(p - self.path[-1]) >= self.res:
            self.path.append(p.copy())
            self.speeds.append(float(np.linalg.norm(self.state.velocity)))
            if len(self.path) > 30000:
                self.path = self.path[::2]
                self.speeds = self.speeds[::2]
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'world'
        t.child_frame_id = f'{self.uav}/base_link_gt'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = \
            (float(x) for x in p)
        q = self.state.quat
        t.transform.rotation.x, t.transform.rotation.y = float(q[0]), float(q[1])
        t.transform.rotation.z, t.transform.rotation.w = float(q[2]), float(q[3])
        self.tf.sendTransform(t)

    def publish(self) -> None:
        arr = MarkerArray()
        if self.dirty_static:
            self.dirty_static = False
            arr.markers += viz.arena_markers(self.course)
            mesh = self.course.obstacle_mesh
            if mesh and mesh.startswith('package://'):
                arr.markers.append(viz.mesh_marker('obstacles', 0, mesh, (0.5, 0.5, 0.52, 0.55)))
        next_id = self.referee.next_gate_id if self.referee is not None else -1
        for k, gate in enumerate(self.course.gates.values()):
            color = viz.GRAY
            if self.missed.get(gate.gate_id):
                color = viz.RED
            elif gate.gate_id == next_id:
                color = viz.YELLOW
            arr.markers += viz.gate_markers(gate, color, k)
        self.cycle += 1
        if len(self.path) >= 2 and self.cycle % 4 == 0:   # the long path at a quarter rate
            arr.markers.append(viz.speed_line('flown', 0, np.asarray(self.path),
                                              np.asarray(self.speeds), self.vmax, width=0.025))
        if self.events:
            arr.markers.append(viz.sphere_list(
                'events', 0, [(e.position.x, e.position.y, e.position.z) for e in self.events],
                [EVENT_COLORS.get(e.type, viz.WHITE) for e in self.events], 0.07))
        if self.state is not None:
            drone = viz.mesh_marker('drone', 0, self.drone_mesh)
            drone.pose.position.x, drone.pose.position.y, drone.pose.position.z = \
                (float(x) for x in self.state.position)
            q = self.state.quat
            drone.pose.orientation.x, drone.pose.orientation.y = float(q[0]), float(q[1])
            drone.pose.orientation.z, drone.pose.orientation.w = float(q[2]), float(q[3])
            drone.scale.x = drone.scale.y = drone.scale.z = self.drone_scale
            arr.markers.append(drone)
            arrow = viz.base_marker('drone', 1, viz.Marker.ARROW)
            p = self.state.position
            v = self.state.velocity
            arrow.points = [viz._point(p), viz._point(p + 0.25 * v)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.015, 0.035, 0.05
            arrow.color = viz.rgba((1.0, 1.0, 1.0, 0.8))
            arr.markers.append(arrow)
        arr.markers.append(viz.text_marker('hud', 0, self.hud_anchor(), self.hud_text(), 0.16))
        self.pub.publish(arr)

    def hud_anchor(self) -> np.ndarray:
        lo, hi = self.course.arena_min, self.course.arena_max
        return np.array([0.5 * (lo[0] + hi[0]), hi[1] + 0.4, hi[2] + 0.3])

    def hud_text(self) -> str:
        race = self.race.state if self.race is not None else '...'
        r = self.referee
        if r is None:
            return f'{race} | referee offline'
        nxt = f'G{r.next_gate_id} (#{r.next_gate_index + 1}/{r.gates_total})' \
            if r.next_gate_id >= 0 else 'landing'
        clear = '' if math.isnan(r.min_clearance) else f'  min clear {r.min_clearance * 100:.0f} cm'
        return (f'{race} / {REFEREE_STATES.get(r.state, "?")}  lap {r.lap}/{r.laps_total}  next {nxt}\n'
                f't = {r.race_time:6.2f} s   v = {r.speed:4.2f} m/s\n'
                f'hits {r.hits}  missed {r.missed_entries}  wrong-way {r.wrong_way}  '
                f'T_adj {r.adjusted_time:6.2f} s{clear}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Visualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:  # rclpy may fail to take a message while the context shuts down
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
