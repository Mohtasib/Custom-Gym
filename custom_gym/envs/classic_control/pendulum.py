import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, reward_type='dense', reward_model=None, angle_threshold=0.15, g=10.0):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.reward_model = reward_model # the type of the reward model, it is either 'CNN' or 'T_CNN'
        self.angle_threshold = angle_threshold # the threshold after which a goal is considered achieved
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

        if 'visual' in self.reward_type:
            global cv2
            import cv2
            from importlib import import_module

            self.frame = None
            self.env_steps = 0
            self.wrong_rewards = 0

            rewardModule = import_module(f'custom_gym.reward_models.{self.reward_model.lower()}_model', package=None)
            rewardClass = getattr(rewardModule, self.reward_model.upper() + '_Model')
            weights = path.join(path.dirname(__file__), f"assets/reward_functions/pendulum/{self.reward_model}_reward_model")
            self.RewardModel = rewardClass()
            self.RewardModel.load_model(weights)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        
        if 'visual' in self.reward_type:
            self.frame = self.render(mode='rgb_array')
            self.env_steps += 1

        reward = self.compute_reward(th, thdot, u)

        return self._get_obs(), reward, False, {}

    def reset(self):
        if 'visual' in self.reward_type:
            # print(' Wrong Reward: {}/{}'.format(self.wrong_rewards, self.env_steps))
            self.env_steps = 0
            self.wrong_rewards = 0

        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.1) # 0.05
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            # self.img = rendering.Image(fname, 1., 1.)
            # self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        # if self.last_u:
            # self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def compute_reward(self, th, thdot, u):
        norm_th = angle_normalize(th)
        costs = norm_th ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        dense_reward = -costs
        sparse_reward = -(norm_th > self.angle_threshold or norm_th < -self.angle_threshold).astype(np.float32)

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

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)