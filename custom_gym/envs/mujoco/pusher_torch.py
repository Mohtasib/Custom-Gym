from os import path
import numpy as np
from gym import utils
from custom_gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable

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

        image_tensor = torch.tensor(img, device=self.device, dtype=torch.float)

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        # Turn the input into a Variable
        inputs = Variable(image_tensor)

        # Predict the class of the image
        output = self.model(inputs)

        return output.data.cpu().numpy()

class PusherEnv_v2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type, distance_threshold):
        self.reward_type = reward_type # the reward type, i.e. 'sparse' or 'dense'
        self.distance_threshold = distance_threshold # the threshold after which a goal is considered achieved
        
        self.state = None
        self.frame = None
        self.env_steps = 0
        self.wrong_rewards = 0

        weights = path.join(path.dirname(__file__), "assets/reward_functions/pusher/FCN_Weights.model")
        self.RewardModel = CNN_Model()
        self.RewardModel.load_model(weights)

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

        # print(' Wrong Reward: {}/{}'.format(self.wrong_rewards, self.env_steps))
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
        reshaped = np.reshape(resized, (num_channels, img_height, img_width)) /255

        reward_img = self.RewardModel.predict(reshaped)[0]
        dense_reward = (2.0*reward_img[1])-1.0
        sparse_reward = np.argmax(reward_img) - 1.0

        true_reward = -(dist > self.distance_threshold).astype(np.float32)

        if sparse_reward != true_reward:
            self.wrong_rewards += 1
            # print(' reward = {}, true reward = {}'.format(sparse_reward, true_reward))

        if self.reward_type == 'sparse':
            return sparse_reward
        else:
            return dense_reward