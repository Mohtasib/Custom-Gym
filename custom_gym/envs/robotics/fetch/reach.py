import os
import numpy as np
from gym import utils
from custom_gym.envs.robotics import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')

class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', reward_model=None, distance_threshold=0.05):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type, reward_model=reward_model)
        utils.EzPickle.__init__(self)

        self.state = None
        if 'visual' in self.reward_type:
            global cv2
            import cv2
            from importlib import import_module

            self.frame = None
            self.env_steps = 0
            self.wrong_rewards = 0

            rewardModule = import_module(f'custom_gym.reward_models.{self.reward_model.lower()}_model', package=None)
            rewardClass = getattr(rewardModule, self.reward_model.upper() + '_Model')
            weights = os.path.join(os.path.dirname(__file__), f"../assets/fetch/reward_functions/reach/{self.reward_model}_reward_model")
            self.RewardModel = rewardClass()
            self.RewardModel.load_model(weights)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        if 'visual' in self.reward_type:
            self.frame = self.render(mode='rgb_array')
            self.env_steps += 1

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs['observation'], reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        # super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()

        if 'visual' in self.reward_type:
            # print(' Wrong Reward: {}/{}'.format(self.wrong_rewards, self.env_steps))
            self.env_steps = 0
            self.wrong_rewards = 0

        obs = self._get_obs()
        return obs['observation']

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        dist = fetch_env.goal_distance(achieved_goal, goal)

        dense_reward = -dist
        sparse_reward = -(dist > self.distance_threshold).astype(np.float32)

        self.state = dist

        if 'visual' in self.reward_type:
            img_width = 100
            img_height = 100
            num_channels = 3
            image = self.frame
            # cv2.imshow('image',image)
            # cv2.waitKey(1)
            image = image[200:1000, 1000:1800]
            resized = cv2.resize(image, (img_width, img_height)) 
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            reshaped = np.reshape(resized, (num_channels, img_height, img_width)) /255

            reward_img = self.RewardModel.predict(reshaped)

            visual_reward_thresholds = 0.8
            
            visual_dense_reward = reward_img[1]-1.0

            visual_sparse_reward = -(reward_img[1] <= visual_reward_thresholds).astype(np.float32)

            if visual_sparse_reward != sparse_reward:
                self.wrong_rewards += 1
                # print(' reward = {}, true reward = {}'.format(dense_reward, dist))

        if self.reward_type == 'visual_sparse':
            return visual_sparse_reward
        elif self.reward_type == 'visual_dense':
            return visual_sparse_reward
        if self.reward_type == 'sparse':
            return sparse_reward
        else:
            return dense_reward