from os import path
import numpy as np
from gym import utils
from custom_gym.envs.mujoco import mujoco_env

from keras.models import load_model
import cv2

import threading

global frame
global reward
global start_computing_rewards
frame = None
reward = None
start_computing_rewards = False

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type, distance_threshold):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        reward = self.compute_reward(-reward_dist)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 0

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0         # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.5         # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def compute_reward(self, dist):
        if self.reward_type == 'sparse':
            return -(dist > self.distance_threshold).astype(np.float32)
        else:
            return -dist

def reward_thread():
    global frame
    global reward
    global start_computing_rewards

    weights = path.join(path.dirname(__file__), "assets/reward_functions/reacher/FCN_Weights.h5")
    RewardModel = load_model(weights)

    while(True):
        if start_computing_rewards:
            reward = RewardModel.predict_on_batch(frame)[0]

class ReacherEnv_v2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type, distance_threshold):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved
        
        self.frame = None
        self.env_steps = 0
        self.wrong_rewards = 0

        t1 = threading.Thread(target=reward_thread, args=[])
        t1.start()

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        self.frame = self.render(mode='rgb_array')
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        reward = self.compute_reward(-reward_dist)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        self.env_steps += 1
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 0

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0         # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.5         # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        # print(' Wrong Reward: {}/{}'.format(self.wrong_rewards, self.env_steps))
        self.env_steps = 0
        self.wrong_rewards = 0
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def compute_reward(self, dist):
        global reward
        global frame
        global start_computing_rewards
        img_width = 100
        img_height = 100
        num_channels = 3
        image = self.frame
        # cv2.imshow('image',image)
        # cv2.waitKey(1)
        resized = cv2.resize(image, (img_width, img_height)) 
        reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels)) /255
        
        frame = reshaped
        start_computing_rewards = True
        
        while(reward is None): pass

        reward_img = reward
        dense_reward = (2.0*reward_img[1])-1.0
        sparse_reward = np.argmax(reward_img) - 1.0
        # sparse_reward = -(dense_reward < 0.005).astype(np.float32)

        true_reward = -(dist > self.distance_threshold).astype(np.float32)

        if sparse_reward != true_reward:
            self.wrong_rewards += 1
            # print(' reward = {}, true reward = {}'.format(sparse_reward, true_reward))

        if self.reward_type == 'sparse':
            return sparse_reward
        else:
            return dense_reward