#!/usr/bin/env python3
"""Offline race-plan sweep: predicts the exact LUS PMM trajectory for combinations of race
parameters and validates each one against the arena and the gate meshes (no simulator needed).

Example (inside the container):
  ros2 run icuas27_race plan_course --start-predictor \
      --speed 2.0 2.5 3.0 --approach 0.25 0.35 --exit 0.25 0.35

Prints a table and writes plan_sweep.json/csv to --out.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys

import numpy as np


def main(argv=None) -> int:
    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from laser_msgs.msg import PoseWithHeading, ReferenceState
    from rclpy.node import Node

    from icuas27_msgs.srv import PredictTrajectory
    from icuas27_race.course import Course
    from icuas27_race.mesh import ObstacleField, load_mesh
    from icuas27_race.planner import (PlanParams, ValidationParams, drop_close_waypoints,
                                      race_waypoints, validate_trajectory)

    bringup = get_package_share_directory('icuas27_bringup')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--course', default=os.path.join(bringup, 'config', 'courses', 'crazyflie_map.yaml'))
    ap.add_argument('--uav-params', default=os.path.join(
        get_package_share_directory('laser_uav_managers'), 'params', 'laser_uavs', 'crazyflie.yaml'))
    ap.add_argument('--pmm-params', default=os.path.join(bringup, 'config', 'lus', 'pmm_crazyflie.yaml'))
    ap.add_argument('--planning-params',
                    default=os.path.join(bringup, 'config', 'lus', 'planning_manager.yaml'))
    ap.add_argument('--start-predictor', action='store_true',
                    help='start icuas27_trajectory/trajectory_predictor with the parameter files')
    ap.add_argument('--speed', type=float, nargs='+', default=[3.0])
    ap.add_argument('--approach', type=float, nargs='+', default=[0.30])
    ap.add_argument('--exit', type=float, nargs='+', default=[0.30])
    ap.add_argument('--heading-mode', default='constant', choices=['constant', 'gate'])
    ap.add_argument('--takeoff-height', type=float, default=1.0,
                    help='hover height above the resting position before the race (m)')
    ap.add_argument('--rest-height', type=float, default=0.015,
                    help='base_link height above the landing base when resting (m)')
    ap.add_argument('--out', default='.')
    args = ap.parse_args(argv)

    proc = None
    if args.start_predictor:
        proc = subprocess.Popen(['ros2', 'run', 'icuas27_trajectory', 'trajectory_predictor', '--ros-args',
                                 '--params-file', args.uav_params, '--params-file', args.pmm_params,
                                 '--params-file', args.planning_params])
    rclpy.init()
    node = Node('plan_course')
    client = node.create_client(PredictTrajectory, '/trajectory_predictor/predict')
    try:
        if not client.wait_for_service(timeout_sec=20.0):
            print('trajectory_predictor service not available', file=sys.stderr)
            return 1
        course = Course.load(args.course)
        mesh = course.obstacle_mesh_path()
        obstacles = None
        if mesh and os.path.exists(mesh):
            obstacles = ObstacleField.from_triangles(
                load_mesh(mesh), {g.name: g.center for g in course.gates.values()})
        start = ReferenceState()
        hover = course.pad_center + np.array([0.0, 0.0, args.rest_height + args.takeoff_height])
        start.pose.position.x, start.pose.position.y, start.pose.position.z = (float(x) for x in hover)
        start.pose.orientation.w = 1.0
        start.use_position = start.use_orientation = True
        start.use_linear_velocity = start.use_angular_velocity = True
        results = []
        for speed, appr, ex in itertools.product(args.speed, args.approach, args.exit):
            params = PlanParams(approach_distance=appr, exit_distance=ex,
                                heading_mode=args.heading_mode)
            wps = drop_close_waypoints(race_waypoints(course, params), hover)
            req = PredictTrajectory.Request()
            req.start = start
            req.speed = speed
            for w in wps:
                pw = PoseWithHeading()
                pw.position.x, pw.position.y, pw.position.z = (float(x) for x in w.position)
                pw.heading = float(w.heading)
                req.waypoints.append(pw)
            fut = client.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
            res = fut.result()
            if res is None or not res.success:
                results.append({'speed': speed, 'approach': appr, 'exit': ex, 'ok': False,
                                'error': res.message if res else 'timeout'})
                continue
            t = np.asarray(res.times)
            p = np.array([[q.x, q.y, q.z] for q in res.positions])
            v = np.array([[q.x, q.y, q.z] for q in res.velocities])
            a = np.array([[q.x, q.y, q.z] for q in res.accelerations])
            rep = validate_trajectory(course, obstacles, p, v, t, ValidationParams())
            results.append({'speed': speed, 'approach': appr, 'exit': ex, 'ok': rep.ok,
                            'duration': rep.duration, 'max_speed': rep.max_speed,
                            'max_accel': float(np.linalg.norm(a, axis=1).max()),
                            'min_clearance': rep.min_clearance,
                            'min_wall_margin': rep.min_wall_margin,
                            'planning_ms': res.planning_ms, 'violations': rep.violations})
        print(f"{'speed':>6} {'appr':>5} {'exit':>5} {'ok':>3} {'dur[s]':>8} {'vmax':>6} "
              f"{'amax':>6} {'clear[m]':>9} {'wall[m]':>8}")
        for r in results:
            if 'duration' not in r:
                print(f"{r['speed']:6.2f} {r['approach']:5.2f} {r['exit']:5.2f}  no  {r['error']}")
                continue
            print(f"{r['speed']:6.2f} {r['approach']:5.2f} {r['exit']:5.2f} {'yes' if r['ok'] else 'NO':>3} "
                  f"{r['duration']:8.2f} {r['max_speed']:6.2f} {r['max_accel']:6.2f} "
                  f"{r['min_clearance']:9.3f} {r['min_wall_margin']:8.3f}")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'plan_sweep.json'), 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2, default=float)
        keys = ['speed', 'approach', 'exit', 'ok', 'duration', 'max_speed', 'max_accel',
                'min_clearance', 'min_wall_margin', 'planning_ms']
        with open(os.path.join(args.out, 'plan_sweep.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for r in results:
                w.writerow([r.get(k, '') for k in keys])
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    sys.exit(main())
