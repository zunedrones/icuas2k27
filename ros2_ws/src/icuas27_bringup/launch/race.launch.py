"""Race nodes only (race manager, referee, predictor, visualiser, telemetry), for a simulation
started separately with sim.launch.py (or the official LUS session)."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from icuas27_race import launch_utils as lu

TRUE = ('1', 'true', 'yes', 'on')


def setup(context):
    env = lu.require_env()

    def get(name):
        return LaunchConfiguration(name).perform(context)

    bringup = lu.share('icuas27_bringup')
    run_dir = os.path.abspath(os.path.expanduser(get('run_dir'))) if get('run_dir') else \
        lu.new_run_dir(get('results_dir'), get('run_name'))
    course_file = os.path.abspath(os.path.expanduser(get('course')))
    lu.write_run_metadata(run_dir, {'course': course_file, 'mode': 'race.launch.py'})
    actions = [LogInfo(msg=f'[icuas27] run directory: {run_dir}')]
    actions += lu.race_node_actions(
        uav=env['UAV_NAME'], uav_type=env['UAV_TYPE'], course_file=course_file,
        race_config=os.path.abspath(get('race_config')), run_dir=run_dir,
        control_manager_file=os.path.abspath(os.path.expanduser(get('control_manager_file'))),
        pmm_planner_file=os.path.abspath(get('pmm_planner_file')),
        planning_manager_file=os.path.abspath(get('planning_manager_file')),
        auto_start=get('auto_start').lower() in TRUE,
        shutdown_on_finish=get('shutdown_on_finish').lower() in TRUE,
        alignment_mode=get('alignment_mode'))
    if get('record').lower() in TRUE:
        actions.append(lu.recorder_action(env['UAV_NAME'], os.path.join(run_dir, 'bag'), False))
    if get('rviz').lower() in TRUE:
        actions.append(Node(package='rviz2', executable='rviz2', output='log',
                            arguments=['-d', os.path.join(bringup, 'rviz', 'race.rviz')],
                            parameters=[{'use_sim_time': True}]))
    return actions


def generate_launch_description():
    bringup = lu.share('icuas27_bringup')
    args = {
        'course': os.path.join(bringup, 'config', 'courses', 'crazyflie_map.yaml'),
        'race_config': os.path.join(bringup, 'config', 'race.yaml'),
        'pmm_planner_file': os.path.join(bringup, 'config', 'lus', 'pmm_crazyflie.yaml'),
        'planning_manager_file': os.path.join(bringup, 'config', 'lus', 'planning_manager.yaml'),
        'control_manager_file': os.path.join(os.path.expanduser('~'), '.icuas27',
                                             'control_manager.yaml'),
        'results_dir': os.path.join(os.path.expanduser('~'), 'results'),
        'run_dir': '', 'run_name': '', 'rviz': 'true', 'record': 'true',
        'auto_start': 'true', 'shutdown_on_finish': 'false', 'alignment_mode': 'ground_truth',
    }
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
                             + [OpaqueFunction(function=setup)])
