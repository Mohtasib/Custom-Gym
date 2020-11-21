from setuptools import setup

setup(name='custom_gym',
      version='0.0.1',
      install_requires=['gym', 'mujoco-py==0.5.7']  # And any other dependencies pepper env needs
)
