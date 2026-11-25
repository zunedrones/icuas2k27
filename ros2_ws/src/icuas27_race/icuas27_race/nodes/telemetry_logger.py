"""Logs the LUS flight-stack signals of a run to CSV for the analytics report.

<run_dir>/telemetry/
  control.csv     control manager diagnostics (commanded reference, NMPC timing, motor thrusts)
  estimation.csv  estimation manager odometry (estimation frame)
  planning.csv    planning manager mission state changes
  motors.csv      normalised motor commands sent to PX4
  px4.csv         PX4 API diagnostics (armed, offboard, preflight)
  race.csv        race manager and referee status
"""

from __future__ import annotations

import csv
import os
from typing import Dict

import rclpy
from laser_msgs.msg import (ApiPx4Diagnostics, MotorSpeed, UavControlDiagnostics,
                            UavPlanningDiagnostics)
from nav_msgs.msg import Odometry
from rclpy.node import Node

from icuas27_msgs.msg import RaceStatus, RefereeStatus
from icuas27_race.ros_utils import (LATCHED_QOS, RELIABLE_QOS, SENSOR_QOS, STATUS_QOS, expand,
                                    stamp_to_sec)


class TelemetryLogger(Node):
    def __init__(self) -> None:
        super().__init__('race_telemetry')
        self.declare_parameter('uav_name', 'uav1')
        self.declare_parameter('run_dir', '')
        uav = self.get_parameter('uav_name').value
        run_dir = self.get_parameter('run_dir').value
        self.out = os.path.join(expand(run_dir), 'telemetry') if run_dir else ''
        self.files: Dict[str, object] = {}
        self.writers: Dict[str, csv.writer] = {}
        if not self.out:
            self.get_logger().warn('run_dir not set: telemetry is not logged')
            return
        os.makedirs(self.out, exist_ok=True)
        self.open('control', ['t', 'rx', 'ref_x', 'ref_y', 'ref_z', 'ref_vx', 'ref_vy', 'ref_vz',
                              'ref_qx', 'ref_qy', 'ref_qz', 'ref_qw', 'ocp_ms', 'iteration_ms',
                              'estimated_mass', 'speed', 'is_fly', 'have_goal', 'u0', 'u1', 'u2',
                              'u3', 'rmse'])
        self.open('estimation', ['t', 'rx', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx_body',
                                 'vy_body', 'vz_body', 'wx', 'wy', 'wz'])
        self.open('planning', ['t', 'rx', 'mission_id', 'state', 'planner', 'waypoint_count',
                               'planning_ms', 'reference_period_ms', 'detail'])
        self.open('motors', ['rx', 'm0', 'm1', 'm2', 'm3'])
        self.open('px4', ['rx', 'armed', 'offboard', 'preflight_ok'])
        self.open('race', ['rx', 'source', 'state', 'elapsed', 'detail'])
        self.last_planning = None
        self.last_px4 = None
        ns = f'/{uav}'
        self.create_subscription(UavControlDiagnostics, f'{ns}/control_manager/diagnostics',
                                 self.on_control, STATUS_QOS)
        self.create_subscription(Odometry, f'{ns}/estimation_manager/estimation',
                                 self.on_estimation, SENSOR_QOS)
        self.create_subscription(UavPlanningDiagnostics, f'{ns}/planning_manager/diagnostics',
                                 self.on_planning, LATCHED_QOS)
        self.create_subscription(MotorSpeed, f'{ns}/control_manager/motor_speed_reference',
                                 self.on_motors, STATUS_QOS)
        self.create_subscription(ApiPx4Diagnostics, f'{ns}/hw_api/diagnostics', self.on_px4,
                                 STATUS_QOS)
        self.create_subscription(RaceStatus, '/race/status', self.on_race, RELIABLE_QOS)
        self.create_subscription(RefereeStatus, '/race/referee/status', self.on_referee,
                                 RELIABLE_QOS)
        self.create_timer(1.0, self.flush)
        self.last_race = None
        self.last_ref = None

    def open(self, name: str, header) -> None:
        fh = open(os.path.join(self.out, f'{name}.csv'), 'w', newline='', encoding='utf-8')
        self.files[name] = fh
        self.writers[name] = csv.writer(fh)
        self.writers[name].writerow(header)

    def rx(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_control(self, m: UavControlDiagnostics) -> None:
        if not m.reference_horizon:
            return
        r = m.reference_horizon[0]
        u = list(m.last_control_input.data)[:4]
        u += [float('nan')] * (4 - len(u))
        self.writers['control'].writerow([
            stamp_to_sec(m.header.stamp), self.rx(),
            r.pose.position.x, r.pose.position.y, r.pose.position.z,
            r.twist.linear.x, r.twist.linear.y, r.twist.linear.z,
            r.pose.orientation.x, r.pose.orientation.y, r.pose.orientation.z, r.pose.orientation.w,
            m.ocp_elapsed_time_ms, m.control_iteration_duration_ms, m.estimated_mass,
            m.current_norm_speed, int(m.is_fly), int(m.have_goal), *u, m.metrics.rmse])

    def on_estimation(self, m: Odometry) -> None:
        p, q = m.pose.pose.position, m.pose.pose.orientation
        v, w = m.twist.twist.linear, m.twist.twist.angular
        self.writers['estimation'].writerow([stamp_to_sec(m.header.stamp), self.rx(), p.x, p.y,
                                             p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z])

    def on_planning(self, m: UavPlanningDiagnostics) -> None:
        key = (m.mission_id, m.state, m.detail)
        if key == self.last_planning:
            return
        self.last_planning = key
        self.writers['planning'].writerow([stamp_to_sec(m.header.stamp), self.rx(), m.mission_id,
                                           m.state, m.planner, m.waypoint_count,
                                           m.planning_duration_ms, m.reference_period_ms, m.detail])

    def on_motors(self, m: MotorSpeed) -> None:
        d = list(m.data)[:4]
        d += [float('nan')] * (4 - len(d))
        self.writers['motors'].writerow([self.rx(), *d])

    def on_px4(self, m: ApiPx4Diagnostics) -> None:
        key = (m.armed, m.offboard_mode, m.preflight_checks_passed)
        if key == self.last_px4:
            return
        self.last_px4 = key
        self.writers['px4'].writerow([self.rx(), int(m.armed), int(m.offboard_mode),
                                      int(m.preflight_checks_passed)])

    def on_race(self, m: RaceStatus) -> None:
        key = (m.state, m.detail)
        if key != self.last_race:
            self.last_race = key
            self.writers['race'].writerow([self.rx(), 'race_manager', m.state, m.elapsed, m.detail])

    def on_referee(self, m: RefereeStatus) -> None:
        key = (m.state, m.detail)
        if key != self.last_ref:
            self.last_ref = key
            self.writers['race'].writerow([self.rx(), 'referee', m.state, m.race_time, m.detail])

    def flush(self) -> None:
        for fh in self.files.values():
            fh.flush()

    def destroy_node(self) -> None:
        for fh in self.files.values():
            fh.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TelemetryLogger()
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
