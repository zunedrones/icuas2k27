"""Replay a recorded race in RViz:

  ros2 launch icuas27_bringup replay.launch.py run_dir:=~/results/<run>   [rate:=0.5]

Plays <run_dir>/bag with its simulation clock; the visualiser rebuilds the course, the drone, the
speed-coloured path, the gate states and the HUD from the recorded ground truth and referee.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from icuas27_race import launch_utils as lu


def setup(context):
    def get(name):
        return LaunchConfiguration(name).perform(context)

    run_dir = os.path.abspath(os.path.expanduser(get('run_dir')))
    bag = os.path.join(run_dir, 'bag')
    if not os.path.isdir(bag):
        raise RuntimeError(f'no bag in {run_dir}')
    course = os.path.join(run_dir, 'config', 'crazyflie_map.yaml')
    if get('course'):
        course = os.path.abspath(os.path.expanduser(get('course')))
    elif not os.path.exists(course):
        course = os.path.join(lu.share('icuas27_bringup'), 'config', 'courses', 'crazyflie_map.yaml')
    uav = os.environ.get('UAV_NAME', 'uav1')
    bringup = lu.share('icuas27_bringup')
    return [
        ExecuteProcess(cmd=['ros2', 'bag', 'play', bag, '--clock', '--rate', get('rate'),
                            '--start-offset', get('start_offset')], output='screen'),
        Node(package='icuas27_race', executable='visualizer', name='race_visualizer', output='log',
             parameters=[os.path.join(bringup, 'config', 'race.yaml'),
                         {'use_sim_time': True, 'course_file': course, 'uav_name': uav,
                          'ground_truth_topic': f'/{uav}/ground_truth'}]),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(bringup, 'rviz', 'race.rviz')],
             parameters=[{'use_sim_time': True}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('run_dir', description='run directory containing bag/'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('start_offset', default_value='0.0'),
        DeclareLaunchArgument('course', default_value=''),
        OpaqueFunction(function=setup),
    ])
