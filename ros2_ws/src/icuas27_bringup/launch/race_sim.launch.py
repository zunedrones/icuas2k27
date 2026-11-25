"""ICUAS 2027 autonomous race in simulation: single, non-interactive entry point.

Starts Gazebo (crazyflie_map), the Crazyflie with PX4 SITL, the LASER UAV System managers, the
race manager (start -> takeoff -> 7 laps -> landing), the referee, telemetry logging, bag
recording and optionally the Gazebo GUI and RViz. Everything is written to the run directory.

  ros2 launch icuas27_bringup race_sim.launch.py                 # GUI + RViz
  ros2 launch icuas27_bringup race_sim.launch.py headless:=true shutdown_on_finish:=true
"""

import os
import shutil

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from icuas27_race import launch_utils as lu
from icuas27_race.course import Course

TRUE = ('1', 'true', 'yes', 'on')


def setup(context):
    env = lu.require_env()
    uav, uav_type = env['UAV_NAME'], env['UAV_TYPE']

    def get(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return get(name).strip().lower() in TRUE

    bringup = lu.share('icuas27_bringup')
    course_file = os.path.abspath(os.path.expanduser(get('course')))
    course = Course.load(course_file)
    run_dir = os.path.abspath(os.path.expanduser(get('run_dir'))) if get('run_dir') else \
        lu.new_run_dir(get('results_dir'), get('run_name'))
    os.makedirs(os.path.join(run_dir, 'config'), exist_ok=True)
    headless = flag('headless')
    gui = flag('gui') and not headless
    rviz = flag('rviz') and not headless
    camera = flag('camera')
    race_config = os.path.abspath(get('race_config'))
    planning_file = os.path.abspath(get('planning_manager_file'))
    pmm_file = os.path.abspath(get('pmm_planner_file'))
    cm_file = lu.write_control_manager_config(
        os.path.join(bringup, 'config', 'lus', 'control_manager.yaml'), course,
        os.path.join(run_dir, 'config', 'control_manager.yaml'))
    for f in (course_file, race_config, planning_file, pmm_file):
        shutil.copy(f, os.path.join(run_dir, 'config', os.path.basename(f)))
    args = {k: get(k) for k in ('world', 'real_time', 'odometry_source', 'control_mode', 'alignment_mode',
                                'auto_start', 'shutdown_on_finish', 'record', 'camera')}
    lu.write_run_metadata(run_dir, {'course': course_file, 'race_config': race_config,
                                    'planning_manager_file': planning_file,
                                    'pmm_planner_file': pmm_file, 'uav': uav,
                                    'uav_type': uav_type, 'args': args})

    actions = [LogInfo(msg=f'[icuas27] run directory: {run_dir}')]
    actions += lu.lus_sim_actions(
        uav=uav, uav_type=uav_type, world=get('world'), gui=gui, real_time=get('real_time'),
        camera=camera, odometry_source=get('odometry_source'), course_file=course_file,
        control_manager_file=cm_file, planning_manager_file=planning_file,
        pmm_planner_file=pmm_file, control_mode=get('control_mode'))
    actions += lu.race_node_actions(
        uav=uav, uav_type=uav_type, course_file=course_file, race_config=race_config,
        run_dir=run_dir, control_manager_file=cm_file, pmm_planner_file=pmm_file,
        planning_manager_file=planning_file, auto_start=flag('auto_start'), shutdown_on_finish=flag('shutdown_on_finish'),
        alignment_mode=get('alignment_mode'), camera=camera)
    if flag('record'):
        actions.append(lu.recorder_action(uav, os.path.join(run_dir, 'bag'), camera))
    if rviz:
        actions.append(Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
                            arguments=['-d', os.path.join(bringup, 'rviz', 'race.rviz')],
                            parameters=[{'use_sim_time': True}]))
    return actions


def generate_launch_description():
    bringup = lu.share('icuas27_bringup')
    defaults = {
        'course': os.path.join(bringup, 'config', 'courses', 'crazyflie_map.yaml'),
        'race_config': os.path.join(bringup, 'config', 'race.yaml'),
        'planning_manager_file': os.path.join(bringup, 'config', 'lus', 'planning_manager.yaml'),
        'pmm_planner_file': os.path.join(bringup, 'config', 'lus', 'pmm_crazyflie.yaml'),
        'results_dir': os.path.join(os.path.expanduser('~'), 'results'),
        'run_dir': '',
        'run_name': '',
        'world': 'crazyflie_map',
        'gui': 'true',
        'rviz': 'true',
        'headless': 'false',
        'camera': 'false',
        'record': 'true',
        'real_time': '1.0',
        'odometry_source': 'ground_truth',
        'control_mode': 'rates',
        'alignment_mode': 'ground_truth',
        'auto_start': 'true',
        'shutdown_on_finish': 'false',
    }
    descriptions = {
        'course': 'course YAML (gates, order, arena, landing base)',
        'race_config': 'parameters of the race nodes',
        'planning_manager_file': 'LUS planning manager parameters (PMM)',
        'pmm_planner_file': 'PMM limits (shared with the trajectory predictor)',
        'results_dir': 'parent directory of the run directories',
        'run_dir': 'explicit run directory (default: <results_dir>/<timestamp>[_<run_name>])',
        'run_name': 'suffix of the run directory name',
        'world': 'Gazebo world in laser_gazebo_resources/worlds',
        'gui': 'start the Gazebo client', 'rviz': 'start RViz',
        'headless': 'no GUI at all (overrides gui/rviz)',
        'camera': 'spawn the AI-deck camera (needs OpenGL rendering)',
        'record': 'record a rosbag2 of the run',
        'real_time': 'PX4_SIM_SPEED_FACTOR',
        'odometry_source': "state estimation input: 'ground_truth' (rulebook) or 'px4'",
        'control_mode': "LUS control: 'individual_thrust' (NMPC+INDI) or 'rates' (NMPC+PX4 rate loop)",
        'alignment_mode': "course-to-estimation frame alignment: ground_truth | identity | start_pose",
        'auto_start': 'start the race automatically when ready (else call /race/start)',
        'shutdown_on_finish': 'stop everything when the race manager finishes',
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=descriptions[k])
         for k, v in defaults.items()] + [OpaqueFunction(function=setup)])
