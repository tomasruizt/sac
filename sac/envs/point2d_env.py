import matplotlib.pyplot as plt
import time
import gym
import numpy as np
from gym.wrappers import TimeLimit


def make_point2d_env():
    return TimeLimit(env=Point2DEnv(), max_episode_steps=50)


class Point2DEnv(gym.Env):
    _INITIAL_POSITION = np.asarray([0, 0])
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self._cur_pos = self._INITIAL_POSITION
        self._plot = None
        self.goal = self._new_goal()

    @staticmethod
    def _new_goal():
        return np.random.uniform(-1, 1, size=(2,))

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (2,) and np.abs(action*0.99).max() <= 1, action
        self._cur_pos = np.clip(a=self._cur_pos + 0.1*action, a_min=-1, a_max=1)
        reward = self.compute_reward(self._cur_pos, self.goal, info=None)
        done = False
        info = dict()
        return self._cur_pos.copy(), reward, done, info

    @staticmethod
    def compute_reward(achieved_goal, desired_goal, info):
        return -np.linalg.norm(np.subtract(achieved_goal, desired_goal), axis=achieved_goal.ndim - 1)

    @staticmethod
    def achieved_goal_from_state(state):
        return state

    def reset(self):
        self._cur_pos = self._INITIAL_POSITION
        self.goal = self._new_goal()
        return self._cur_pos.copy()

    def render(self, mode='human'):
        if self._plot is None:
            plt.ion() if mode == "human" else plt.ioff()
            fig, ax = plt.subplots()
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
            ax.plot(*zip(*corners))
            path_collection = ax.scatter(*self._cur_pos, c="red")
            goal_path_coll = ax.scatter(*self.goal, c="green")
            self._plot = fig, path_collection, goal_path_coll
        fig, path_collection, goal_path_coll = self._plot
        path_collection.set_offsets(self._cur_pos)
        goal_path_coll.set_offsets(self.goal)
        fig.canvas.draw()

        if mode == "rgb_array":
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            return data.reshape((height, width, 3))

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        np.random.seed(seed)


if __name__ == '__main__':
    use_imgs = False
    env = Point2DEnv()
    img = env.render() if not use_imgs else env.render("rgb_array")
    if use_imgs:
        plt.ion()
        fig, ax = plt.subplots()
        obj = ax.imshow(img)
        fig.canvas.draw()
        fig.canvas.flush_events()
    while True:
        env.reset()
        for _ in range(30):
            print(env.step(env.action_space.sample())[1])
            time.sleep(0.1)
            img = env.render() if not use_imgs else env.render("rgb_array")
            if use_imgs:
                obj.set_data(img)
                fig.canvas.draw()
                fig.canvas.flush_events()
