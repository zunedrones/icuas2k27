from setuptools import find_packages, setup

package_name = 'icuas27_race'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zunedrones',
    maintainer_email='zunedrones@example.com',
    description='Autonomous drone racing for the ICUAS 2027 UAV Competition on the LASER UAV System.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'race_manager = icuas27_race.nodes.race_manager:main',
            'referee = icuas27_race.nodes.referee:main',
            'gt_odometry_bridge = icuas27_race.nodes.gt_odometry_bridge:main',
            'visualizer = icuas27_race.nodes.visualizer:main',
            'telemetry_logger = icuas27_race.nodes.telemetry_logger:main',
            'camera_view = icuas27_race.nodes.camera_view:main',
            'race_report = icuas27_race.analytics.report:main',
            'race_compare = icuas27_race.analytics.compare:main',
            'extract_course = icuas27_race.tools.extract_course:main',
            'plan_course = icuas27_race.tools.plan_course:main',
        ],
    },
)
