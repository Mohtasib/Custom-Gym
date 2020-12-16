from __future__ import absolute_import
from gym.envs.registration import register


# Classic
# --------------------------------------------------

register(
    id='CustomPendulumDense-v1',
    entry_point='custom_gym.envs.classic_control:PendulumEnv',
    kwargs = {
            'reward_type': 'dense',
            'angle_threshold': 0.15,
        },
    max_episode_steps=200,
)

register(
    id='CustomPendulumSparse-v1',
    entry_point='custom_gym.envs.classic_control:PendulumEnv',
    kwargs = {
            'reward_type': 'sparse',
            'angle_threshold': 0.15,
        },
    max_episode_steps=200,
)

register(
    id='CustomPendulumVisualDense-v1',
    entry_point='custom_gym.envs.classic_control:PendulumEnv_v2',
    kwargs = {
            'reward_type': 'dense',
            'angle_threshold': 0.15,
        },
    max_episode_steps=200,
)

register(
    id='CustomPendulumVisualSparse-v1',
    entry_point='custom_gym.envs.classic_control:PendulumEnv_v2',
    kwargs = {
            'reward_type': 'sparse',
            'angle_threshold': 0.15,
        },
    max_episode_steps=200,
)

# Mujoco
# --------------------------------------------------

register(
    id='CustomReacherDense-v1',
    entry_point='custom_gym.envs.mujoco:ReacherEnv',
    kwargs = {
            'reward_type': 'dense',
            'distance_threshold': 0.005,
        },
    max_episode_steps=50,
)

register(
    id='CustomReacherSparse-v1',
    entry_point='custom_gym.envs.mujoco:ReacherEnv',
    kwargs = {
            'reward_type': 'sparse',
            'distance_threshold': 0.02,
        },
    max_episode_steps=50,
)

register(
    id='CustomReacherVisualSparse-v1',
    entry_point='custom_gym.envs.mujoco:ReacherEnv_v2',
    kwargs = {
            'reward_type': 'sparse',
            'distance_threshold': 0.02,
        },
    max_episode_steps=50,
)

register(
    id='CustomPusherDense-v1',
    entry_point='custom_gym.envs.mujoco:PusherEnv',
    kwargs = {
            'reward_type': 'dense',
            'distance_threshold': 0.005,
        },
    max_episode_steps=100,
)

register(
    id='CustomPusherSparse-v1',
    entry_point='custom_gym.envs.mujoco:PusherEnv',
    kwargs = {
            'reward_type': 'sparse',
            'distance_threshold': 0.01,
        },
    max_episode_steps=100,
)

# --------------------------------------------------
register(
    id='CustomReacherPixel-v1',
    entry_point='custom_gym.envs.mujoco:ReacherEnvPixel',
    kwargs = {
            'reward_type': 'sparse',
            'distance_threshold': 0.005,
            'obs_shape': [100, 100]
        },
    max_episode_steps=100,
    reward_threshold=0.0,
)
# --------------------------------------------------
register(
    id='CustomPusherPixel-v1',
    entry_point='custom_gym.envs.mujoco:PusherEnvPixel',
    kwargs = {
            'obs_shape': [100, 100]
        },
    max_episode_steps=100,
    reward_threshold=0.0,
)
# --------------------------------------------------
register(
    id='CustomHalfCheetahPixel-v1',
    entry_point='custom_gym.envs.mujoco:HalfCheetahEnvPixel',
    kwargs = {
            'obs_shape': [100, 100]
        },
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
# --------------------------------------------------


# Pepper Aldebaran Robot
# --------------------------------------------------

register(
    id='PepperSimDIRECTDense-v1',
    entry_point='custom_gym.envs.pepper:PepperSimEnv',
    kwargs = {
            'reward_type': 'dense',
            'distance_threshold': 0.08,
            'image_shape': [100, 100, 3],
            'render_mode': 'DIRECT'
        },
    max_episode_steps=500,
)
# --------------------------------------------------
register(
    id='PepperSimGUIDense-v1',
    entry_point='custom_gym.envs.pepper:PepperSimEnv',
    kwargs = {
            'reward_type': 'dense',
            'distance_threshold': 0.08,
            'image_shape': [100, 100, 3],
            'render_mode': 'GUI'
        },
    max_episode_steps=500,
)
# --------------------------------------------------
register(
    id='PepperSimDIRECTSparse-v1',
    entry_point='custom_gym.envs.pepper:PepperSimEnv',
    kwargs = {
            'reward_type': 'sparse',
            'distance_threshold': 0.08,
            'image_shape': [100, 100, 3],
            'render_mode': 'DIRECT'
        },
    max_episode_steps=500,
)
# --------------------------------------------------
register(
    id='PepperSimGUISparse-v1',
    entry_point='custom_gym.envs.pepper:PepperSimEnv',
    kwargs = {
            'reward_type': 'sparse',
            'distance_threshold': 0.08,
            'image_shape': [100, 100, 3],
            'render_mode': 'GUI'
        },
    max_episode_steps=500,
)
# --------------------------------------------------