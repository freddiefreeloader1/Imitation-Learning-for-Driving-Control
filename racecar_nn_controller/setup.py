from setuptools import setup
import os
import glob

package_name = 'racecar_nn_controller'

# Get all files in the models directory
models_dir = os.path.join(package_name, 'models')
model_files = glob.glob(models_dir + '/*')  # Glob will list all files in the directory

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', model_files),  # Automatically include all model files
    ],
    install_requires=['setuptools', 'torch'],
    zip_safe=True,
    maintainer='Emre Gursoy',
    maintainer_email='mustafa.gursoy@epfl.ch',
    description='ROS2 package for racecar neural network controller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = racecar_nn_controller.controller_node:main',
            'controller_node_transformer = racecar_nn_controller.controller_node_transformer:main',
            'pure_pursuit_controller = racecar_nn_controller.pure_pursuit_controller:main',
        ],
    },
)
