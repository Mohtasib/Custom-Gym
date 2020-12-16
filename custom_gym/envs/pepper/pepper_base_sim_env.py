import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from custom_gym.envs.pepper.Environment import *
from custom_gym.envs.pepper.Robot import *
from qibullet import Camera
import math
import cv2
import threading
import time
import os
from PIL import Image
from custom_gym.envs.pepper import pepper_kinematics as pk
from os import path
import random
import pybullet as p

class PepperBaseSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, image_shape=(100,100,3), render_mode='DIRECT'):
        self.render_mode = render_mode
        self.image_shape = image_shape
        self.box_pos = [0.26, 0, 0.8]
        self.box_ort = [0, 0, -0.7071, 0.7071]
        self.box_final_pos = None
        self.box_final_ort = None
        self.aux_data = None
        self.reward = 0
        self.done = False
        self.fractionMaxSpeed = 1
        self.joints = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw", "RHand", "LShoulderPitch", "LShoulderRoll", "LElbowRoll", "LElbowYaw", "LWristYaw", "LHand"]
        self.Init_QiBullet_Env()

        self.img = self.pepper.getCameraFrame(self.CameraHandler)
        [self.ImageHeight,self.ImageWidth,_] = self.img.shape

        self.low_angles = np.zeros(12)
        self.high_angles = np.zeros(12)
        for i in range(0,12):
            self.low_angles[i], self.high_angles[i] = self.robot.getJointRangeValue(self.joints[i])  

        self.action_space = spaces.Box(
            low=self.low_angles[:6],
            high=self.high_angles[:6], shape=(6,),
            dtype=np.float32
        )

        # TODO: you need to define the observation space
        # self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4))
        high = np.inf*np.ones(13)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.seed()

    def step(self, action):
        self.do_action(action)
        obs = self._get_obs()
        self.done = self.check_DONE()
        self.reward = self.compute_reward()
        return (obs, self.reward, self.done, {})

    def _get_image(self):
        return self.pepper.getCameraFrame(self.CameraHandler)
        
    def _format_img(self, img):
        img_width = self.image_shape[0]
        img_height = self.image_shape[1]
        num_channels = self.image_shape[2]

        resized = cv2.resize(img, (img_width, img_height)) 
        reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels)) /255
        return reshaped

    def _calculate_pose(self, angles):
        current_position, rotation_matrix = pk.right_arm_get_position(angles[:6])
        current_orientation = pk.rotation_matrix_to_rpy(rotation_matrix)
        return np.concatenate((current_position[:3], current_orientation))

    def estimate_aux_data(self):
        angles = self._get_angles()

        pose = self._calculate_pose(angles)
        velocities = self._get_velocities()[:6]
        hand = np.array(round(angles[5], 0))

        aux_data = np.hstack((pose, velocities, hand))
        
        return aux_data

    def _get_obs(self):
        boxfinalPos = p.getBasePositionAndOrientation(self.box)
        self.box_final_pos = np.array(boxfinalPos[0])
        self.box_final_ort = np.array(boxfinalPos[1])
        ort = abs(self.box_final_ort - self.box_ort)
        pos = abs(self.box_final_ort - self.box_ort)
        if any(item >= 0.2 for item in ort) or (pos[0] >= 0.2) or (pos[1] >= 0.2):  
            p.resetBasePositionAndOrientation(self.box, self.box_pos, self.box_ort)
            time.sleep(0.2)

        self.aux_data = self.estimate_aux_data()

        return self.aux_data

    def compute_reward(self):
        """
        Calculate the reward.
        Implement this in each subclass.
        """
        raise NotImplementedError

    def check_DONE(self):        
        return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_goal(self):
        """
        This method is called to set a goal for the currrent episode.
        """
        pass

    def _get_angles(self):
        angles = self.pepper.getAnglesPosition(self.joints)
        return np.array(angles)

    def _get_velocities(self):
        velocities = self.pepper.getAnglesVelocity(self.joints)
        return np.array(velocities)

    def do_action(self, action):
        actions = [ action[0],
                    action[1],
                    action[2],
                    action[3],
                    action[4],
                    action[5],
                    action[0],
                    0.0 - action[1],
                    0.0 - action[2],
                    0.0 - action[3],
                    0.0 - action[4],
                    action[5]]

        self.pepper.setAngles(self.joints, actions, self.fractionMaxSpeed)
        time.sleep(0.05)

    def _go_init_pos(self):
        # needed for resetting
        self.robot.resetAngles()
        self.robot.setPercentageSpeed(1)
        self.pepper.setAngles(list(self.robot.joint_parameters.keys()), list(self.robot.joint_parameters.values()), self.robot.getPercentageSpeed())
        self.robot.waitEndMove(self.pepper)
        p.resetBasePositionAndOrientation(self.box, self.box_pos, self.box_ort)
        time.sleep(2)
        self.time_previous = 0.0
        self.angles_previous = self._get_angles()

    def Init_QiBullet_Env(self):
        if self.render_mode == 'GUI':
            self.environment = Environment_GUI()
        else:
            self.environment = Environment_DIRECT()
        self.pepper = self.environment.createPepper(translation=[0, 0, 0])
        self.robot = Robot(self.pepper)
        # Resolution choices: {160x120 (K_QQVGA), 320x240 (K_QVGA), 640x480 (K_VGA),
        #                      320x180 (K_QQ720p), 640x360 (K_Q720p), 1280x720 (K_720p)}
        self.CameraHandler = self.pepper.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM, resolution=Camera.K_QQVGA)
        fname = path.join(path.dirname(__file__), "Obj_files/table/table.urdf")
        self.table = p.loadURDF(fname, [0.7, 0, 0], [0, 0, math.sqrt(0.5), math.sqrt(0.5)])
        fname = path.join(path.dirname(__file__), "Obj_files/box/box.urdf")
        self.box = p.loadURDF(fname, self.box_pos, self.box_ort)

    def reset(self):
        self._go_init_pos()
        self.done = False
        self.reward = 0
        return self._get_obs()

    def render(self, mode='human'):
        pass
