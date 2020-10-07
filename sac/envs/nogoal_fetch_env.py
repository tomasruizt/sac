from typing import Union

import gym
import time

import numpy as np
from gym.envs.robotics import FetchReachEnv, FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit, FilterObservation, FlattenObservation


def wrap_fetch_env(fetch_env: Union[FetchReachEnv, FetchPickAndPlaceEnv]):
    env = FilterObservation(env=fetch_env, filter_keys=["observation", "achieved_goal"])
    bounded_space = gym.spaces.Box(low=np.array([1.2, 0, 0.4]), high=np.array([1.6, 1.2, 0.9]))
    env.observation_space = gym.spaces.Dict(spaces=dict(
        achieved_goal=bounded_space,
        observation=env.observation_space["observation"])
    )
    return TimeLimit(env=FlattenObservation(env), max_episode_steps=1000)


def make_nogoal_fetchreach_env():
    return wrap_fetch_env(FetchReachEnv())


def make_nogoal_fetchpickandplace_env():
    return wrap_fetch_env(FetchPickAndPlaceEnv())


if __name__ == '__main__':
    env = make_nogoal_fetchpickandplace_env()
    assert isinstance(env.reset(), np.ndarray)
    while True:
        env.step(env.action_space.sample())
        img = env.render()  # mode="rgb_array" gives and OpenGL error on my setup
        time.sleep(1/20)
