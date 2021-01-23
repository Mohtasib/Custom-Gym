import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import cv2
import threading

import torch
import torch.nn as nn
from torch.autograd import Variable

global frame
global reward
global start_computing_rewards
frame = None
reward = None
start_computing_rewards = False

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, reward_type, angle_threshold, g=10.0):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
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

        reward = self.compute_reward(th, thdot, u)

        return self._get_obs(), reward, False, {}

    def reset(self):
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
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def compute_reward(self, th, thdot, u):
        norm_th = angle_normalize(th)
        # print(' reward = {}'.format(-(norm_th > self.angle_threshold or norm_th < -self.angle_threshold).astype(np.float32)))
        if self.reward_type == 'sparse':
            return -(norm_th > self.angle_threshold or norm_th < -self.angle_threshold).astype(np.float32)
        else:
            costs = norm_th ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
            return -costs

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit2 = Unit(in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=128, out_channels=64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.unit5 = Unit(in_channels=64, out_channels=32)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv = nn.Conv2d(in_channels=32,kernel_size=3,out_channels=num_classes,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(   self.unit1, self.pool1, 
                                    self.unit2, self.pool2,
                                    self.unit3, self.pool3,
                                    self.unit4, self.pool4,
                                    self.unit5, self.pool5,
                                    self.conv, self.relu, self.avgpool, self.flatten, self.softmax)

    def forward(self, input):
        output = self.net(input)
        # output = output.view(-1,128)
        # output = self.fc(output)
        return output

class CNN_Model():
    def __init__(self):
        # CUDA for PyTorch
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        self.model = None

    def load_model(self, path):
        checkpoint = torch.load(path)
        model = SimpleNet()
        model.load_state_dict(checkpoint)
        model.to(self.device)
        self.model = model

    def predict(self, img):
        self.model.eval()

        image_tensor = torch.Tensor(img)

        image_tensor = torch.tensor(img, device=self.device, dtype=torch.float)

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        # Turn the input into a Variable
        inputs = Variable(image_tensor)

        # Predict the class of the image
        output = self.model(inputs)

        return output.data.cpu().numpy()

def reward_thread():
    global frame
    global reward
    global start_computing_rewards

    weights = path.join(path.dirname(__file__), "assets/reward_functions/pendulum/FCN_Weights.model")
    RewardModel = CNN_Model()
    RewardModel.load_model(weights)

    while(True):
        if start_computing_rewards:
            reward = RewardModel.predict(frame)[0]

class PendulumEnv_v2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, reward_type, angle_threshold, g=10.0):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.angle_threshold = angle_threshold # the threshold after which a goal is considered achieved
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.frame = None

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
        
        t1 = threading.Thread(target=reward_thread, args=[])
        t1.start()

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
        self.render(mode='rgb_array')

        reward = self.compute_reward(th, thdot, u)

        return self._get_obs(), reward, False, {}

    def reset(self):
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

        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        self.frame = self.viewer.render(return_rgb_array=True)
        return self.frame

    def _get_frame(self, mode='rgb_array'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
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
        global reward
        global frame
        global start_computing_rewards
        img_width = 100
        img_height = 100
        num_channels = 3
        image = self.frame
        resized = cv2.resize(image, (img_width, img_height)) 
        reshaped = np.reshape(resized, (num_channels, img_height, img_width)) /255
        
        frame = reshaped
        start_computing_rewards = True
        
        while(reward is None): pass

        reward_img = reward
        dense_reward = (2.0*reward_img[1])-1.0
        sparse_reward = np.argmax(reward_img) - 1.0
        # sparse_reward = -(dense_reward < 0.8).astype(np.float32)

        norm_th = angle_normalize(th)
        true_reward = -(norm_th > self.angle_threshold or norm_th < -self.angle_threshold).astype(np.float32)

        # if sparse_reward != true_reward:
            # print(' reward = {}, true reward = {}'.format(sparse_reward, true_reward))

        if self.reward_type == 'sparse':
            return sparse_reward
        else:
            return dense_reward

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)