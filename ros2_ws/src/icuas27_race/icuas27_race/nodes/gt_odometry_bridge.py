"""Ground-truth odometry for the LUS estimation manager.

The rulebook states that in simulation the state estimation uses the simulator ground truth.
This node turns the Gazebo P3D ground truth into odometry with the same conventions as the
LUS PX4 hardware API (laser_uav_hw_api px4_api_node, /uav1/hw_api/odometry):

  * pose relative to the pose captured when the drone first rests on the base,
  * frame_id "<uav>/fcu", child_frame_id "<uav>/hw_api_odometry",
  * linear and angular twist in the body (FLU) frame.

Unlike the hardware API, only the initial yaw is removed (not roll/pitch), so the frame stays
gravity aligned. The estimation manager is launched with its odometry input remapped here.
Also republishes the world-frame ground truth (offsets undone) on ~/world for the race nodes.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from icuas27_race.course import Course
from icuas27_race.geometry import (make_transform, matrix_to_quat_xyzw, rot_z,
                                   yaw_of_matrix)
from icuas27_race.ros_utils import RELIABLE_QOS, SENSOR_QOS, GroundTruthAdapter, expand


class GroundTruthBridge(Node):
    def __init__(self) -> None:
        super().__init__('gt_odometry_bridge')
        self.declare_parameter('uav_name', 'uav1')
        self.declare_parameter('course_file', '')
        self.declare_parameter('ground_truth_topic', '/uav1/ground_truth')
        self.declare_parameter('output_topic', '/uav1/race/gt_odometry')
        self.declare_parameter('world_topic', '/uav1/race/ground_truth_world')
        self.declare_parameter('rest_speed', 0.02)        # m/s
        self.declare_parameter('rest_angular_speed', 0.05)  # rad/s
        self.declare_parameter('rest_time', 1.0)          # s at rest before the origin is fixed
        self.declare_parameter('pose_covariance', 1e-6)
        self.declare_parameter('twist_covariance', 1e-6)

        self.uav = self.get_parameter('uav_name').value
        course_file = self.get_parameter('course_file').value
        if not course_file:
            raise RuntimeError('course_file parameter is required (spawn pose = ground-truth offset)')
        course = Course.load(expand(course_file))
        self.adapter = GroundTruthAdapter(course.start_position, course.start_rpy)
        self.rest_speed = float(self.get_parameter('rest_speed').value)
        self.rest_w = float(self.get_parameter('rest_angular_speed').value)
        self.rest_time = float(self.get_parameter('rest_time').value)
        self.pose_cov = float(self.get_parameter('pose_covariance').value)
        self.twist_cov = float(self.get_parameter('twist_covariance').value)

        self.origin = None          # T_world_odom (4, 4) once captured
        self.rest_since = None
        # Reliable publishers: the LUS estimation manager subscribes with reliable QoS (a
        # best-effort publisher would be silently ignored); reliable also serves best effort.
        self.pub_odom = self.create_publisher(
            Odometry, self.get_parameter('output_topic').value, RELIABLE_QOS)
        self.pub_world = self.create_publisher(
            Odometry, self.get_parameter('world_topic').value, RELIABLE_QOS)
        self.create_subscription(Odometry, self.get_parameter('ground_truth_topic').value,
                                 self.on_ground_truth, SENSOR_QOS)
        self.get_logger().info('waiting for the drone to rest on the base to fix the odometry origin')

    def on_ground_truth(self, msg: Odometry) -> None:
        state = self.adapter.convert(msg)
        self.publish_world(msg, state)
        if self.origin is None:
            at_rest = (np.linalg.norm(state.velocity) < self.rest_speed and
                       np.linalg.norm(state.angular_velocity) < self.rest_w)
            if not at_rest:
                self.rest_since = None
                return
            if self.rest_since is None:
                self.rest_since = state.stamp
                return
            if state.stamp - self.rest_since < self.rest_time:
                return
            yaw0 = yaw_of_matrix(state.rotation)
            self.origin = make_transform(rot_z(yaw0), state.position)
            self.get_logger().info(
                f'odometry origin fixed at world p=({state.position[0]:.3f}, '
                f'{state.position[1]:.3f}, {state.position[2]:.3f}) yaw={math.degrees(yaw0):.2f} deg')
        r0 = self.origin[:3, :3]
        p_rel = r0.T @ (state.position - self.origin[:3, 3])
        r_rel = r0.T @ state.rotation
        q_rel = matrix_to_quat_xyzw(r_rel)
        v_body = state.rotation.T @ state.velocity

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = f'{self.uav}/fcu'
        out.child_frame_id = f'{self.uav}/hw_api_odometry'
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = \
            (float(x) for x in p_rel)
        out.pose.pose.orientation.x, out.pose.pose.orientation.y, \
            out.pose.pose.orientation.z, out.pose.pose.orientation.w = (float(x) for x in q_rel)
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = \
            (float(x) for x in v_body)
        out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z = \
            (float(x) for x in state.angular_velocity)
        for i in range(6):
            out.pose.covariance[7 * i] = self.pose_cov
            out.twist.covariance[7 * i] = self.twist_cov
        self.pub_odom.publish(out)

    def publish_world(self, msg: Odometry, state) -> None:
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'world'
        out.child_frame_id = f'{self.uav}/base_link_gt'
        p = state.position
        q = state.quat
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = \
            (float(x) for x in p)
        out.pose.pose.orientation.x, out.pose.pose.orientation.y, \
            out.pose.pose.orientation.z, out.pose.pose.orientation.w = (float(x) for x in q)
        # World-frame linear velocity here (documented on the topic), body angular rate.
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = \
            (float(x) for x in state.velocity)
        out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z = \
            (float(x) for x in state.angular_velocity)
        self.pub_world.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthBridge()
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
