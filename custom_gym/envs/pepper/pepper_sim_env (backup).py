
import time
import numpy as np
from gym import spaces
from custom_gym.envs.pepper.pepper_base_sim_env import PepperBaseSimEnv
import pybullet as p

class PepperSimEnv(PepperBaseSimEnv):
    def __init__(self, reward_type, distance_threshold, image_shape, render_mode):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.image_shape = image_shape
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8)
        super().__init__(self.image_shape, self.render_mode)

    def compute_reward(self):
        boxfinalPos = p.getBasePositionAndOrientation(self.box)
        self.box_final_pos = np.array(boxfinalPos[0])
        dist = self.box_final_pos[2] - self.box_pos[2]

        if self.reward_type == 'sparse':
            return -(dist < self.distance_threshold).astype(np.float32)
        else:
            return dist