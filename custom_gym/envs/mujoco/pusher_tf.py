from os import path
import numpy as np
from gym import utils
from custom_gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib
import cv2
from keras.models import load_model

class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type, distance_threshold):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved
        self.state = None
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def step(self, a):
        # vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()

        self.state = -reward_dist
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = self.compute_reward(-reward_dist)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = -1
    #     self.viewer.cam.distance = 4.0
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1         # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.9        # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def compute_reward(self, dist):
        if self.reward_type == 'sparse':
            return -(dist > self.distance_threshold).astype(np.float32)
        else:
            return -dist

class PusherEnv_v2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type, distance_threshold):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved
        
        self.state = None
        self.frame = None
        self.env_steps = 0
        self.wrong_rewards = 0

        weights = path.join(path.dirname(__file__), "assets/reward_functions/pusher/FCN_Weights.h5")
        self.RewardModel = load_model(weights)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def step(self, a):
        # vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()

        self.state = -reward_dist
        self.frame = self.render(mode='rgb_array')
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = self.compute_reward(-reward_dist)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        self.env_steps += 1
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = -1
    #     self.viewer.cam.distance = 4.0
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1         # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.9        # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)

        print(' Wrong Reward: {}/{}'.format(self.wrong_rewards, self.env_steps))
        self.env_steps = 0
        self.wrong_rewards = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def compute_reward(self, dist):
        img_width = 100
        img_height = 100
        num_channels = 3
        image = self.frame
        # cv2.imshow('image',image)
        # cv2.waitKey(1)
        resized = cv2.resize(image, (img_width, img_height)) 
        reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels)) /255

        reward_img = self.RewardModel.predict_on_batch(reshaped)[0]
        
        visual_reward_thresholds = 0.8

        if reward_img[1] >= visual_reward_thresholds:
            dense_reward = (reward_img[1] - visual_reward_thresholds)/(1 - visual_reward_thresholds)
        else:
            dense_reward = (reward_img[1] / visual_reward_thresholds) - 1

        sparse_reward = -(reward_img[1] <= visual_reward_thresholds).astype(np.float32)

        true_reward = -(dist > self.distance_threshold).astype(np.float32)

        if sparse_reward != true_reward:
            self.wrong_rewards += 1
            # print(' reward = {}, true reward = {}'.format(sparse_reward, true_reward))

        if self.reward_type == 'sparse':
            return sparse_reward
        else:
            return dense_reward