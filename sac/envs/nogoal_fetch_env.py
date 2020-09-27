import gym
import time

import numpy as np
from gym.envs.robotics import FetchReachEnv
from gym.wrappers import TimeLimit, FilterObservation, FlattenObservation


def make_nogoal_fetchreach_env():
    env = FilterObservation(env=FetchReachEnv(), filter_keys=["observation", "achieved_goal"])
    bounded_space = gym.spaces.Box(low=np.array([1.2, 0, 0.4]), high=np.array([1.6, 1.2, 0.9]))
    env.observation_space = gym.spaces.Dict(spaces=dict(
        achieved_goal=bounded_space,
        observation=env.observation_space["observation"])
    )
    return TimeLimit(env=FlattenObservation(env), max_episode_steps=1000)


if __name__ == '__main__':
    env = make_nogoal_fetchreach_env()
    assert isinstance(env.reset(), np.ndarray)
    while True:
        env.step(env.action_space.sample())
        img = env.render()  # mode="rgb_array" gives and OpenGL error on my setup
        time.sleep(1/20)
