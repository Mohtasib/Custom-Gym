
import os
from os import path
import numpy as np
import math
import random
import time
from pkg_resources import parse_version

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cv2
import threading

from qibullet import Camera
from qibullet import PepperVirtual
import pybullet as p
import pybullet_data

from custom_gym.envs.pepper import pepper_kinematics as pk
from custom_gym.envs.pepper.Environment import Environment_GUI, Environment_DIRECT

ACTION_MIN = np.array([-0.6, -0.6, -2.0857, 0.0087, -1.8239, 0.0])
ACTION_MAX = np.array([0.6, -0.0087, 2.0857, 1.36136, 1.8239, 1.0])
BOX_POS = [0.26, 0, 0.8]
BOX_ORT = [0, 0, -0.7071, 0.7071]

def rescale_linear(arr, arr_min, arr_max, new_min, new_max):
    """Rescale an arrary linearly."""
    new_arr = np.zeros_like(arr)
    for i in range(arr.shape[-1]):
        m = (new_max[i] - new_min[i]) / (arr_max[i] - arr_min[i])
        b = new_min[i] - m * arr_min[i]
        new_arr[i] = m * arr[i] + b
    return new_arr

class PepperBaseSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                image_shape=(100,100,3),
                urdfRoot=pybullet_data.getDataPath(),
                actionRepeat=1,
                renders=False,
                isDiscrete=False,
                maxSteps=5):
        self.image_shape = image_shape
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._maxSteps = maxSteps
        self.box_pos = BOX_POS
        self.box_ort = BOX_ORT
        self.box_final_pos = None
        self.box_final_ort = None
        self.obs = None
        self.img = None
        self.aux_data = None
        self.reward = 0
        self.done = False
        self.fractionMaxSpeed = 1
        self.joints = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand"]

        self._init_env()
        self.seed()
        self.reset()

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(12)
        else:
            low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 
            self.action_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                                            dtype=np.uint8)
        
        self.obs = self._get_obs()

        self.viewer = None
    
    def _init_env(self):
        if self._renders:
            self.environment = Environment_GUI()
        else:
            self.environment = Environment_DIRECT()
        self.pepper = self.environment.createPepper(translation=[0, 0, 0])
        # Resolution choices: {160x120 (K_QQVGA), 320x240 (K_QVGA), 640x480 (K_VGA),
        #                      320x180 (K_QQ720p), 640x360 (K_Q720p), 1280x720 (K_720p)}
        self.CameraHandler = self.pepper.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM, resolution=Camera.K_QQVGA)
        fname = path.join(path.dirname(__file__), "Obj_files/table/table.urdf")
        self.table = p.loadURDF(fname, [0.7, 0, 0], [0, 0, math.sqrt(0.5), math.sqrt(0.5)])
        fname = path.join(path.dirname(__file__), "Obj_files/box/box.urdf")
        self.box = p.loadURDF(fname, self.box_pos, self.box_ort)

    def reset(self):
        self.pepper.setAngles(self.joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], self.fractionMaxSpeed)
        
        p.resetBasePositionAndOrientation(self.box, self.box_pos, self.box_ort)
        time.sleep(2)

        self._envStepCounter = 0
        self.obs = self._get_obs()
        self.done = False
        self.reward = 0
        return self.obs

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._apply_action(action)
        self.obs = self._get_obs()
        self.done = self._check_termination()
        self.reward = self._compute_reward()
        return (self.obs, self.reward, self.done, {})

    def _get_image(self):
        return self.pepper.getCameraFrame(self.CameraHandler)
        
    def _format_img(self, img):
        img_width = self.image_shape[0]
        img_height = self.image_shape[1]
        num_channels = self.image_shape[2]

        resized = cv2.resize(img, (img_width, img_height)) 
        reshaped = np.reshape(resized, (img_width, img_height, num_channels)) #/255.0
        return reshaped

    def _angles_to_pose(self, angles):
        current_position, rotation_matrix = pk.right_arm_get_position(angles[:6])
        current_orientation = pk.rotation_matrix_to_rpy(rotation_matrix)
        return np.concatenate((current_position[:3], current_orientation))

    def _pose_to_angles(self, angles, pose):
        target_angles =  pk.right_arm_set_position(angles, pose[:3], pose[3:])
        # Here we check if the distance can not donverged, then we return the current angles:
        if target_angles is None:
            return angles
        return target_angles

    def _estimate_aux_data(self):
        angles = self._get_angles()

        pose = self._angles_to_pose(angles)
        velocities = self._get_velocities()[:6]
        hand = np.array(round(angles[5], 0))

        aux_data = np.hstack((pose, velocities, hand))
        
        return aux_data

    def _get_obs(self):
        img = self._get_image()
        self.img = self._format_img(img)

        self.aux_data = self._estimate_aux_data()

        # return (self.img, self.aux_data)
        return self.img

    def _compute_reward(self):
        """
        Calculate the reward.
        Implement this in each subclass.
        """
        raise NotImplementedError

    def _check_termination(self):
        if (self._envStepCounter > self._maxSteps):
            self.obs = self._get_obs()
            return True        
        return False

    def _set_goal(self):
        """
        This method is called to set a goal for the currrent episode.
        """
        pass

    def _get_angles(self):
        angles = self.pepper.getAnglesPosition(self.joints[:6])
        return np.array(angles)

    def _get_velocities(self):
        velocities = self.pepper.getAnglesVelocity(self.joints[:6])
        return np.array(velocities)

    def _get_real_action(self, action):
        current_angles = self._get_angles()
        if (self._isDiscrete):
            deltas = [0.01, 0.01, 0.01, 0.01, 0.01, 1]
            d1 = [-deltas[0], deltas[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action] # delta RShoulderPitch
            d2 = [0, 0, -deltas[1], deltas[1], 0, 0, 0, 0, 0, 0, 0, 0][action] # delta RShoulderRoll
            d3 = [0, 0, 0, 0, -deltas[2], deltas[2], 0, 0, 0, 0, 0, 0][action] # delta RElbowYaw
            d4 = [0, 0, 0, 0, 0, 0, -deltas[3], deltas[3], 0, 0, 0, 0][action] # delta RElbowRoll
            d5 = [0, 0, 0, 0, 0, 0, 0, 0, -deltas[4], deltas[4], 0, 0][action] # delta RWristYaw
            dh = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -deltas[5], deltas[5]][action] # delta RHand
            
            delta_angles = [d1, d2, d3, d4, d5]

            if dh == -1: hand = 0.0
            elif dh == 1: hand = 1.0
            else: hand = current_angles[5]

            action_angles = np.concatenate(((current_angles[:5] + delta_angles), np.array([hand])))
        else:
            low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            action_angles = rescale_linear(action, low, high, ACTION_MIN, ACTION_MAX)

        realAction = [  action_angles[0],
                        action_angles[1],
                        action_angles[2],
                        action_angles[3],
                        action_angles[4],
                        action_angles[5],
                        action_angles[0],
                        0.0 - action_angles[1],
                        0.0 - action_angles[2],
                        0.0 - action_angles[3],
                        0.0 - action_angles[4],
                        action_angles[5]]
        return realAction

    def _apply_action(self, action):
        realAction = self._get_real_action(action)

        for _ in range(self._actionRepeat):
            self.pepper.setAngles(self.joints, realAction, self.fractionMaxSpeed)
            if self._check_termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        return self.img

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step