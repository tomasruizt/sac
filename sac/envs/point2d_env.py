import matplotlib.pyplot as plt
import time
import gym
import numpy as np
from gym.wrappers import TimeLimit

gym.register(id="Point2D-v0", entry_point="sac.envs.point2d_env:make_point2d_env")


def make_point2d_env():
    return TimeLimit(env=Point2DEnv())


class Point2DEnv(gym.Env):
    _INITIAL_POSITION = np.asarray([0, 0])

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self._cur_pos = self._INITIAL_POSITION
        self._plot = None

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (2,) and np.abs(action*0.99).max() <= 1, action
        self._cur_pos = np.clip(a=self._cur_pos + 0.1*action, a_min=-1, a_max=1)
        reward = 0
        done = False
        info = dict()
        return self._cur_pos.copy(), reward, done, info

    def reset(self):
        self._cur_pos = self._INITIAL_POSITION
        return self._cur_pos.copy()

    def render(self, mode='human'):
        if self._plot is None:
            plt.ion() if mode == "human" else plt.ioff()
            fig, ax = plt.subplots()
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
            ax.plot(*zip(*corners))
            path_collection = ax.scatter(*self._cur_pos, c="red")
            self._plot = fig, path_collection
        fig, path_collection = self._plot
        path_collection.set_offsets(self._cur_pos)
        fig.canvas.draw()
        #fig.canvas.flush_events()

        if mode == "rgb_array":
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            return data.reshape((height, width, 3))

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)


if __name__ == '__main__':
    use_imgs = True
    env = Point2DEnv()
    img = env.render() if not use_imgs else env.render("rgb_array")
    if use_imgs:
        plt.ion()
        fig, ax = plt.subplots()
        obj = ax.imshow(img)
        fig.canvas.draw()
        fig.canvas.flush_events()
    while True:
        env.step(env.action_space.sample())
        time.sleep(0.1)
        img = env.render() if not use_imgs else env.render("rgb_array")
        if use_imgs:
            obj.set_data(img)
            fig.canvas.draw()
            fig.canvas.flush_events()
