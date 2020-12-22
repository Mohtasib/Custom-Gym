
import time
import numpy as np
from gym import spaces
from custom_gym.envs.pepper.pepper_base_sim_env import PepperBaseSimEnv
import pybullet as p

class PepperSimEnv(PepperBaseSimEnv):
    def __init__(   self, 
                    reward_type,
                    distance_threshold,
                    image_shape,
                    actionRepeat=1,
                    renders=False,
                    isDiscrete=False,
                    maxSteps=500):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.image_shape = image_shape
        self.actionRepeat = actionRepeat
        self.renders = renders
        self.isDiscrete = isDiscrete
        self.maxSteps = maxSteps

        super().__init__(   image_shape=self.image_shape, 
                            actionRepeat=self.actionRepeat,
                            renders=self.renders,
                            isDiscrete=self.isDiscrete,
                            maxSteps=self.maxSteps,
                            )

    def _compute_reward(self):
        boxfinalPos = p.getBasePositionAndOrientation(self.box)
        self.box_final_pos = np.array(boxfinalPos[0])
        dist = self.box_final_pos[2] - self.box_pos[2]

        if self.reward_type == 'sparse':
            return -float(dist < self.distance_threshold)
        else:
            return dist