import gym

from .gym_env import GymEnv
from .multigoal import MultiGoalEnv

gym.register(id="Point2D-v0", entry_point="sac.envs.point2d_env:make_point2d_env")
gym.register(id="NoGoalFetchReach-v0", entry_point="sac.envs.nogoal_fetch_env:make_nogoal_fetchreach_env")
gym.register(id="NoGoalFetchPickAndPlace-v0", entry_point="sac.envs.nogoal_fetch_env:make_nogoal_fetchpickandplace_env")
