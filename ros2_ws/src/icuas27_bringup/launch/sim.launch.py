"""Simulation stack only (Gazebo, Crazyflie, PX4 SITL, LUS managers), with the race
configuration. Use with race.launch.py to start the race nodes separately, or to fly manually:

  ros2 launch icuas27_bringup sim.launch.py
  ros2 service call /uav1/hw_api/arm std_srvs/srv/Trigger {}
  ros2 service call /uav1/control_manager/takeoff std_srvs/srv/Trigger {}
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from icuas27_race import launch_utils as lu
from icuas27_race.course import Course

TRUE = ('1', 'true', 'yes', 'on')


def setup(context):
    env = lu.require_env()

    def get(name):
        return LaunchConfiguration(name).perform(context)

    bringup = lu.share('icuas27_bringup')
    course_file = os.path.abspath(os.path.expanduser(get('course')))
    course = Course.load(course_file)
    cfg_dir = os.path.abspath(os.path.expanduser(get('config_dir')))
    cm_file = lu.write_control_manager_config(
        os.path.join(bringup, 'config', 'lus', 'control_manager.yaml'), course,
        os.path.join(cfg_dir, 'control_manager.yaml'))
    return [LogInfo(msg=f'[icuas27] control manager parameters: {cm_file}')] + lu.lus_sim_actions(
        uav=env['UAV_NAME'], uav_type=env['UAV_TYPE'], world=get('world'),
        gui=get('gui').lower() in TRUE, real_time=get('real_time'),
        camera=get('camera').lower() in TRUE, odometry_source=get('odometry_source'),
        course_file=course_file, control_manager_file=cm_file,
        planning_manager_file=os.path.abspath(get('planning_manager_file')),
        pmm_planner_file=os.path.abspath(get('pmm_planner_file')),
        control_mode=get('control_mode'))


def generate_launch_description():
    bringup = lu.share('icuas27_bringup')
    args = {
        'course': os.path.join(bringup, 'config', 'courses', 'crazyflie_map.yaml'),
        'planning_manager_file': os.path.join(bringup, 'config', 'lus', 'planning_manager.yaml'),
        'pmm_planner_file': os.path.join(bringup, 'config', 'lus', 'pmm_crazyflie.yaml'),
        'config_dir': os.path.join(os.path.expanduser('~'), '.icuas27'),
        'world': 'crazyflie_map', 'gui': 'true', 'camera': 'false', 'real_time': '1.0',
        'odometry_source': 'ground_truth',
        'control_mode': 'rates',
    }
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
                             + [OpaqueFunction(function=setup)])
