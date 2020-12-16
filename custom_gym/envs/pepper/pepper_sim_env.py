
import numpy as np
from custom_gym.envs.pepper.pepper_base_sim_env import PepperBaseSimEnv

class PepperSimEnv(PepperBaseSimEnv):
    def __init__(self, reward_type, distance_threshold, image_shape, render_mode):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.image_shape = image_shape
        self.render_mode = render_mode
        super().__init__(self.image_shape, self.render_mode)

    def compute_reward(self):

        dist = self.box_final_pos[2] - self.box_pos[2]

        if self.reward_type == 'sparse':
            return -(dist < self.distance_threshold).astype(np.float32)
        else:
            return dist