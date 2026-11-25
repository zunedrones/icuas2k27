"""Referee node: scores the run from the Gazebo ground truth (independent of the flight stack)
and logs everything needed by the analytics report.

Outputs in <run_dir>/referee/: summary.json, events.csv, passes.csv, laps.csv, trajectory.csv.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from icuas27_msgs.msg import GateEvent, RaceEvent, RefereeStatus
from icuas27_race.course import Course
from icuas27_race.mesh import ObstacleField, load_mesh
from icuas27_race.referee_core import (EVENT_CODES, FINISHED, INVALID, RUNNING, STATE_CODES,
                                       RefereeCore, RefereeEvent, RefereeParams)
from icuas27_race.ros_utils import (EVENTS_QOS, RELIABLE_QOS, SENSOR_QOS, GroundTruthAdapter,
                                    expand, stamp_to_sec)

TRAJ_FIELDS = ['stamp', 'race_time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw',
               'wx', 'wy', 'wz', 'speed', 'clearance']


class RefereeNode(Node):
    def __init__(self) -> None:
        super().__init__('race_referee')
        self.declare_parameter('course_file', '')
        self.declare_parameter('ground_truth_topic', '/uav1/ground_truth')
        self.declare_parameter('run_dir', '')
        self.declare_parameter('status_rate', 10.0)
        self.declare_parameter('pre_start_log', 2.0)   # s of trajectory kept before the start
        self.declare_parameter('post_finish_log', 3.0)  # s logged after finish/invalidation
        defaults = RefereeParams()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)

        course_file = self.get_parameter('course_file').value
        if not course_file:
            raise RuntimeError('course_file parameter is required')
        self.course = Course.load(expand(course_file))
        params = RefereeParams(**{k: type(v)(self.get_parameter(k).value)
                                  for k, v in vars(defaults).items()})
        obstacles = None
        mesh_path = self.course.obstacle_mesh_path()
        if mesh_path and os.path.exists(mesh_path):
            centers = {g.name: g.center for g in self.course.gates.values()}
            obstacles = ObstacleField.from_triangles(load_mesh(mesh_path), centers)
            self.get_logger().info(f'obstacle mesh: {mesh_path} '
                                   f'({sum(len(c.triangles) for c in obstacles.clusters)} triangles)')
        else:
            self.get_logger().warn(f'obstacle mesh not found ({mesh_path}): contacts are not detected')
        self.core = RefereeCore(self.course, params, obstacles)
        self.adapter = GroundTruthAdapter(self.course.start_position, self.course.start_rpy)

        run_dir = self.get_parameter('run_dir').value
        self.out_dir = os.path.join(expand(run_dir), 'referee') if run_dir else ''
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
        self.pre_log = float(self.get_parameter('pre_start_log').value)
        self.post_log = float(self.get_parameter('post_finish_log').value)
        self.buffer: Deque[list] = deque()
        self.samples: List[list] = []
        self.written = False
        self.end_stamp: Optional[float] = None
        self.last_stamp = 0.0
        self.last_state = None

        self.pub_status = self.create_publisher(RefereeStatus, '/race/referee/status', RELIABLE_QOS)
        self.pub_events = self.create_publisher(GateEvent, '/race/referee/events', EVENTS_QOS)
        self.create_subscription(Odometry, self.get_parameter('ground_truth_topic').value,
                                 self.on_ground_truth, SENSOR_QOS)
        self.create_subscription(RaceEvent, '/race/events', self.on_race_event, EVENTS_QOS)
        rate = float(self.get_parameter('status_rate').value)
        self.create_timer(1.0 / rate, self.publish_status)
        self.get_logger().info(
            f'referee ready: {self.course.name}, {self.course.laps} laps x '
            f'{self.course.gates_per_lap} gates, order {self.course.order}')

    # ------------------------------------------------------------------ callbacks
    def on_race_event(self, msg: RaceEvent) -> None:
        stamp = stamp_to_sec(msg.header.stamp)
        if msg.type == RaceEvent.START and self.core.state not in (RUNNING, FINISHED, INVALID):
            self.core.start(stamp)
            # keep only the configured pre-start window
            self.samples = [s for s in self.buffer if s[0] >= stamp - self.pre_log]
            self.buffer.clear()
            self.get_logger().info(f'race started at t={stamp:.3f} (rest height '
                                   f'{self.core.rest_height if self.core.rest_height is not None else float("nan"):.3f} m)')
        elif msg.type == RaceEvent.ABORTED and self.core.state == RUNNING:
            self.core.abort(stamp, f'aborted by the race manager: {msg.detail}')
            self.get_logger().warn(f'run aborted: {msg.detail}')

    def on_ground_truth(self, msg: Odometry) -> None:
        st = self.adapter.convert(msg)
        self.last_stamp = st.stamp
        events = self.core.update(st.stamp, st.position, st.velocity, st.rotation)
        row = [st.stamp, self.core.race_time(st.stamp) if self.core.t_start else float('nan'),
               *st.position, *st.velocity, *st.quat, *st.angular_velocity,
               float(np.linalg.norm(st.velocity)),
               self.core.last_clearance if math.isfinite(self.core.last_clearance) else float('nan')]
        if self.core.t_start is None:
            self.buffer.append(row)
            while self.buffer and self.buffer[0][0] < st.stamp - self.pre_log:
                self.buffer.popleft()
        elif not self.written:
            self.samples.append(row)
        for ev in events:
            self.publish_event(ev)
            text = (f'[{ev.race_time:7.3f}s] {ev.type} gate={ev.gate_id} idx={ev.gate_index} '
                    f'lap={ev.lap} off=({ev.offset_y:+.3f},{ev.offset_z:+.3f}) '
                    f'v={ev.speed:.2f} {ev.detail}')
            # rclpy forbids one call site from logging at different severities.
            if ev.type == 'PASS':
                self.get_logger().info(text)
            else:
                self.get_logger().warn(text)
        if self.core.state in (FINISHED, INVALID) and self.end_stamp is None:
            self.end_stamp = st.stamp
            s = self.core.summary()
            self.get_logger().info(
                f'run {s["state"]}: T={s["elapsed_time"]:.3f}s penalties={s["penalty_time"]:.1f}s '
                f'T_adj={s["adjusted_time"]} reason="{s["reason"]}"')
        if self.end_stamp is not None and not self.written and \
                st.stamp - self.end_stamp >= self.post_log:
            self.write_results()

    # ------------------------------------------------------------------ outputs
    def publish_event(self, ev: RefereeEvent) -> None:
        msg = GateEvent()
        msg.header.stamp.sec = int(ev.stamp)
        msg.header.stamp.nanosec = int(round((ev.stamp - int(ev.stamp)) * 1e9)) % 1000000000
        msg.header.frame_id = 'world'
        msg.type = EVENT_CODES[ev.type]
        msg.gate_id = int(ev.gate_id)
        msg.gate_index = int(ev.gate_index)
        msg.lap = int(ev.lap)
        msg.race_time = float(ev.race_time)
        msg.position.x, msg.position.y, msg.position.z = (float(x) for x in ev.position)
        msg.offset_y = float(ev.offset_y)
        msg.offset_z = float(ev.offset_z)
        msg.speed = float(ev.speed)
        msg.clearance = float(ev.clearance)
        msg.detail = ev.detail
        self.pub_events.publish(msg)

    def publish_status(self) -> None:
        c = self.core
        msg = RefereeStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.state = STATE_CODES[c.state]
        msg.race_time = float(c.race_time(self.last_stamp))
        msg.next_gate_index = int(c.next_index)
        msg.next_gate_id = int(c.sequence[c.next_index]) if c.next_index < c.gates_total else -1
        msg.lap = int(c.lap())
        msg.laps_total = int(self.course.laps)
        msg.gates_passed = len(c.passes)
        msg.gates_total = int(c.gates_total)
        msg.hits = int(c.hits)
        msg.missed_entries = int(c.missed_entries)
        msg.wrong_way = int(c.wrong_way)
        msg.penalty_time = float(c.penalty_time)
        msg.adjusted_time = float(msg.race_time + c.penalty_time)
        msg.speed = float(c.last_speed)
        msg.min_clearance = float(c.min_clearance) if math.isfinite(c.min_clearance) else float('nan')
        msg.detail = c.reason
        self.pub_status.publish(msg)

    def write_results(self) -> None:
        if self.written:
            return
        self.written = True
        if not self.out_dir:
            return
        summary = self.core.summary()
        summary['course'] = self.course.to_dict()
        summary['referee_params'] = vars(self.core.p)
        with open(os.path.join(self.out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2, default=float)
        with open(os.path.join(self.out_dir, 'trajectory.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(TRAJ_FIELDS)
            w.writerows(self.samples)
        ev_fields = ['type', 'stamp', 'race_time', 'gate_id', 'gate_index', 'lap', 'x', 'y', 'z',
                     'offset_y', 'offset_z', 'speed', 'clearance', 'detail']
        with open(os.path.join(self.out_dir, 'events.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(ev_fields)
            for e in self.core.events:
                w.writerow([e.type, e.stamp, e.race_time, e.gate_id, e.gate_index, e.lap,
                            *e.position, e.offset_y, e.offset_z, e.speed, e.clearance, e.detail])
        pass_fields = ['index', 'gate_id', 'lap', 'stamp', 'race_time', 'offset_y', 'offset_z',
                       'speed', 'clearance', 'approach_angle', 'retry']
        with open(os.path.join(self.out_dir, 'passes.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(pass_fields)
            for p in self.core.passes:
                w.writerow([p[k] for k in pass_fields])
        laps = self.core.lap_times()
        with open(os.path.join(self.out_dir, 'laps.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['lap', 'lap_time', 'flying_lap_time'])
            for i, t in enumerate(laps['by_last_gate']):
                fly = laps['flying'][i] if i < len(laps['flying']) else ''
                w.writerow([i + 1, t, fly])
        self.get_logger().info(f'results written to {self.out_dir}')

    def destroy_node(self) -> None:
        if not self.written and self.core.t_start is not None:
            if self.core.state == RUNNING:
                self.core.abort(self.last_stamp, 'node shut down before the end of the run')
            self.write_results()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RefereeNode()
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
