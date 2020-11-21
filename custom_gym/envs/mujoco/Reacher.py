import numpy as np
from gym import utils
from custom_gym.envs.mujoco import mujoco_env

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