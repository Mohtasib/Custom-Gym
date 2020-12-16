import numpy as np
from gym import utils
from custom_gym.envs.mujoco import mujoco_env_pixel

import mujoco_py
from mujoco_py.mjlib import mjlib

from skimage import color
from skimage import transform

class ReacherEnvPixel(mujoco_env_pixel.MujocoEnvPixel, utils.EzPickle):
    def __init__(self, reward_type, distance_threshold, obs_shape):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved
        self.obs_shape = obs_shape
        self.memory = np.empty([self.obs_shape[0], self.obs_shape[1], 4],dtype=np.uint8)
        utils.EzPickle.__init__(self)
        mujoco_env_pixel.MujocoEnvPixel.__init__(self, 'reacher.xml', 2)

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
        data = self._get_viewer().get_image()
        rawByteImg = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(rawByteImg, dtype=np.uint8)
        img = np.reshape(tmp, [height, width, 3])
        img = np.flipud(img) # 500x500x3
        gray = color.rgb2gray(img) # convert to gray
        gray_resized = transform.resize(gray,(self.obs_shape[0], self.obs_shape[1])) # resize
        # update memory buffer
        # self.memory[1:,:,:] = self.memory[0:3,:,:]
        self.memory[:,:,1:] = self.memory[:,:,0:3]
        # self.memory[0,:,:] = gray_resized
        self.memory[:,:,0] = gray_resized*255

        return self.memory

    def compute_reward(self, dist):
        if self.reward_type == 'sparse':
            return -(dist > self.distance_threshold).astype(np.float32)
        else:
            return -dist