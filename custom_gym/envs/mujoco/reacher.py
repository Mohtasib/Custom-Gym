from os import path
import numpy as np
from gym import utils
from custom_gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='dense', reward_model=None, distance_threshold=0.02):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.reward_model = reward_model # the type of the reward model, it is either 'CNN' or 'T_CNN'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved

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
            weights = path.join(path.dirname(__file__), f"assets/reward_functions/reacher/{self.reward_model}_reward_model")
            self.RewardModel = rewardClass()
            self.RewardModel.load_model(weights)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        if 'visual' in self.reward_type:
            self.frame = self.render(mode='rgb_array')
            self.env_steps += 1

        self.state = -reward_dist
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

        if 'visual' in self.reward_type:
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
        dense_reward = -dist
        sparse_reward = -(dist > self.distance_threshold).astype(np.float32)

        if 'visual' in self.reward_type:
            img_width = 100
            img_height = 100
            num_channels = 3
            image = self.frame
            # cv2.imshow('image',image)
            # cv2.waitKey(1)
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