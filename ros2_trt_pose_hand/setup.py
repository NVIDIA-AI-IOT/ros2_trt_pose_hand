from setuptools import setup
from glob import glob
import os

package_name = 'ros2_trt_pose_hand'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ak-nv',
    maintainer_email='ameykulkarni@nvidia.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand-pose-estimation = ros2_trt_pose_hand.live_hand_pose:main',
        ],
    },
)
