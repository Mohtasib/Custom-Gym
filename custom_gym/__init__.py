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
# ----------------------------------------

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
            'distance_threshold': 0.005,
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
            'distance_threshold': 0.005,
        },
    max_episode_steps=100,
)

# --------------------------------------------------
register(
    id='CustomPusherPixel-v0',
    entry_point='custom_gym.envs.mujoco:PusherEnvPixel',
    max_episode_steps=100,
    reward_threshold=0.0,
)
# --------------------------------------------------
register(
    id='CustomHalfCheetahPixel-v1',
    entry_point='custom_gym.envs.mujoco:HalfCheetahEnvPixel',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
# --------------------------------------------------


