"""Drone's-eye view: draws the course gates projected into the AI-deck camera image.

With the ground-truth pose and the camera intrinsics, every gate opening (and frame) is
projected into the image; the expected gate is highlighted and its visibility (fraction of the
opening inside the image, apparent area) is logged to <run_dir>/camera/visibility.csv. This is
the reference a camera gate detector will be evaluated against.

Publishes /race/camera/overlay (sensor_msgs/Image, bgr8).
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from icuas27_msgs.msg import RefereeStatus
from icuas27_race.camera_model import PinholeCamera
from icuas27_race.course import Course
from icuas27_race.ros_utils import (RELIABLE_QOS, SENSOR_QOS, GroundTruthAdapter, expand,
                                    stamp_to_sec)


class CameraView(Node):
    def __init__(self) -> None:
        super().__init__('race_camera_view')
        self.declare_parameter('course_file', '')
        self.declare_parameter('run_dir', '')
        self.declare_parameter('ground_truth_topic', '/uav1/ground_truth')
        self.declare_parameter('image_topic', '/uav1/camera/ai_deck/image_raw')
        self.declare_parameter('camera_info_topic', '/uav1/camera/ai_deck/camera_info')
        # AI-deck lens in the Crazyflie body frame (crazyflie.sdf.jinja: link z = 0.015 m,
        # sensor x = 0.029 m, no rotation).
        self.declare_parameter('camera_offset', [0.029, 0.0, 0.015])
        course = Course.load(expand(self.get_parameter('course_file').value))
        self.course = course
        self.adapter = GroundTruthAdapter(course.start_position, course.start_rpy)
        self.offset = np.asarray(self.get_parameter('camera_offset').value, dtype=float)
        self.bridge = CvBridge()
        self.camera: Optional[PinholeCamera] = None
        self.state = None
        self.referee: Optional[RefereeStatus] = None
        self.writer = None
        run_dir = self.get_parameter('run_dir').value
        if run_dir:
            out = os.path.join(expand(run_dir), 'camera')
            os.makedirs(out, exist_ok=True)
            self.file = open(os.path.join(out, 'visibility.csv'), 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['t', 'next_gate', 'visible_fraction', 'area_px', 'distance'])
        self.pub = self.create_publisher(Image, '/race/camera/overlay', 5)
        self.create_subscription(Odometry, self.get_parameter('ground_truth_topic').value,
                                 lambda m: setattr(self, 'state', self.adapter.convert(m)),
                                 SENSOR_QOS)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value,
                                 self.on_info, SENSOR_QOS)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image,
                                 SENSOR_QOS)
        self.create_subscription(RefereeStatus, '/race/referee/status',
                                 lambda m: setattr(self, 'referee', m), RELIABLE_QOS)

    def on_info(self, msg: CameraInfo) -> None:
        if self.camera is None and msg.k[0] > 0:
            self.camera = PinholeCamera.from_camera_info(msg, self.offset)
            self.get_logger().info(f'camera {msg.width}x{msg.height} fx={msg.k[0]:.1f}')

    def on_image(self, msg: Image) -> None:
        if self.camera is None or self.state is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        st = self.state
        next_id = self.referee.next_gate_id if self.referee is not None else -1
        best = None
        for gate in self.course.gates.values():
            for inner, base_color in ((False, (90, 90, 90)), (True, (200, 200, 200))):
                corners = gate.corners(inner=inner)
                uv, front = self.camera.project(
                    self.camera.world_to_optical(corners, st.rotation, st.position))
                if not front.all():
                    continue
                color = (0, 200, 255) if gate.gate_id == next_id else base_color
                pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], True, color, 2 if gate.gate_id == next_id else 1)
                if inner:
                    c = np.round(uv.mean(axis=0)).astype(int)
                    cv2.putText(img, f'G{gate.gate_id}', (int(c[0]) - 8, int(c[1]) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            if gate.gate_id == next_id:
                corners = gate.corners(inner=True)
                uv, front = self.camera.project(
                    self.camera.world_to_optical(corners, st.rotation, st.position))
                frac, area = self.camera.polygon_visibility(uv, front)
                best = (frac, area, float(np.linalg.norm(gate.center - st.position)))
        if best is not None:
            cv2.putText(img, f'next G{next_id}: {best[0] * 100:.0f}% visible, {best[2]:.2f} m',
                        (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
            if self.writer is not None:
                self.writer.writerow([stamp_to_sec(msg.header.stamp), next_id, best[0], best[1],
                                      best[2]])
        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)

    def destroy_node(self) -> None:
        if self.writer is not None:
            self.file.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraView()
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
