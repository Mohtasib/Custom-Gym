from setuptools import setup

setup(name='custom_gym',
      version='1.0.0',
      install_requires=['gym==0.21.0', 
			'pybullet==3.0.8',
			'qibullet==1.4.3',
			'mujoco-py==2.0.2.13',
			'torch==1.8.0']  # And any other dependencies pepper env needs
)
