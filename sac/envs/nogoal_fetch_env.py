import time

import numpy as np
from gym.envs.robotics import FetchReachEnv
from gym.wrappers import TimeLimit, FilterObservation, FlattenObservation


def make_nogoal_fetchreach_env():
    no_desired_goal_env = FilterObservation(env=FetchReachEnv(), filter_keys=["observation", "achieved_goal"])
    return TimeLimit(env=FlattenObservation(no_desired_goal_env), max_episode_steps=1000)


if __name__ == '__main__':
    env = make_nogoal_fetchreach_env()
    assert isinstance(env.reset(), np.ndarray)
    while True:
        env.step(env.action_space.sample())
        img = env.render()  # mode="rgb_array" gives and OpenGL error on my setup
        time.sleep(1/20)
