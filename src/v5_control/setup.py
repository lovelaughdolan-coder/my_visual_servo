from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'v5_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装 launch 文件
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('..', '..', 'launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hxy',
    maintainer_email='hxy@todo.todo',
    description='视觉伺服控制包（PBVS/IBVS）及手眼标定工具',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pbvs_controller = v5_control.pbvs_controller:main',
            'pbvs_controller_eye_in_hand = v5_control.pbvs_controller_eye_in_hand:main',
            'ibvs_controller_eye_in_hand = v5_control.ibvs_controller_eye_in_hand:main',
            'ibvs_yolo_controller = v5_control.ibvs_yolo_controller:main',
            'hand_eye_calibrator_eye_in_hand = v5_control.hand_eye_calibrator_eye_in_hand:main',
            'robot_init = v5_control.robot_init:main',
        ],
    },
)
