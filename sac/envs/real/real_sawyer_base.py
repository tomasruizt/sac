from __future__ import absolute_import
import time
import numpy as np
import scipy
import tf # ros transformations library
import random
import os
import sys

import rospy

from intera_interface import CHECK_VERSION, limb, RobotEnable

from rllab import spaces
from rllab.misc.overrides import overrides
from cached_property import cached_property

import config

from rllab.envs.base import Env

ALL_JOINTS = (
    'right_l0',
    'right_l1',
    'right_l2',
    'right_l3',
    'right_l4',
    'right_l5',
    'right_l6',
    'right_hand'
)

ALL_LINKS = (
    'right_l0',
    'right_l1',
    'right_l2',
    'right_l3',
    'right_l4',
    'right_l5',
    'right_l6',
    'right_hand',
    'right_gripper_base',
    'right_gripper'
)

NEUTRAL = np.array([ 0.28625879, -0.06652148, -1.45274023,  1.87639258,  1.47005664,
        1.47084668, -0.87656152])
MAX_TORQUES = 0.5 * np.array([8, 12, 6, 5, 4, 3, 6]) # torque limits for each joint

class SawyerEnv(Env):
    def __init__(self,
                 action_cost_coeff=1.0,
                 joint_mask=None,
                 reset_every_n=1,
                 include_xpos=False,
                 include_pose=False):
        super(SawyerEnv, self).__init__()

        """Boolean mask indicating which joints should be controlled"""
        if joint_mask is None:
            joint_mask = [True, True, True, True, True, True, True]
        self._joint_mask = joint_mask
        
        self._action_cost_coeff = action_cost_coeff

        """Actions are torques applied to each joint enabled in the joint mask"""
        self._Da = sum(self._joint_mask) 
        """Observations include joint angles and angular velocities for each joint"""
        self._Do = 2 * self._Da

        """
        If include_xpos is True, observations include the Cartesian coordinates of 
        the end effector, right_hand
        """
        if include_xpos:
            self._Do += 3

        """
        If include_pos is True, observaions include the orientation of the end effector
        in Euler angles
        """
        if include_pose:
            self._Do += 3

        self._include_xpos = include_xpos
        self._include_pose = include_pose

        self._num_iter = 0
        self._reset_every_n = reset_every_n

    def initialize(self, use_remote_name=False):
        """
        This function allows us to create instances of the env
        without interfacing with ROS. This is used when
        remote envs need to access the environment (and actual
        robot) to sample actions, while local processes also
        need the environment to access the env spec but do not 
        need to actually control the robot.
        :param use_remote_name: Should be True if being initialized in a remote environment
        :return: None
        """
        if use_remote_name:
            rospy.init_node("sawyer_interface_remote")
        else:
            rospy.init_node("sawyer_interface")
        rs = RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        if not init_state:
            rs.enable()
        rospy.on_shutdown(self.clean_shutdown)
        self._arm = limb.Limb('right')
        self._kinematics = sawyer_kinematics('right')
        self._rate = rospy.Rate(20)

    @cached_property
    @overrides
    def action_space(self):
        masked_limits = self.MAX_TORQUES[self._joint_mask]
        return spaces.Box(-masked_limits, masked_limits)

    @cached_property
    @overrides
    def observation_space(self):

        lim = np.zeros(self._Do) * 1E6
        return spaces.Box(-lim, lim)
    
    """
    Resets the environment and returns the new initial observation for the next rollout.
    """
    @overrides
    def reset(self):
        if self._num_iter % self._reset_every_n == 0:
            self.reset_arm_to_neutral()
        self._num_iter += 1
        time.sleep(1)
        return self.get_observation()

    """
    Executed at each time step, this sets the joint torques for the given action
    and returns information dependent on the task. At each step, it should return
    the next observation, the reward received, whether the episode has terminated, 
    and an optionally a dictionary containing information about the environment for 
    logging purposes.
    """
    @overrides
    def step(self, action):
        """
        If some joints are not enabled, we automatically set torques to
        keep them at the angle specified in the NEUTRAL position.
        """
        if not all(self._joint_mask):
            angle = self.get_joint_angles()
            action_all = self._get_pd_torques(angle - NEUTRAL)
            action_all[self._joint_mask] = action
        else:
            action_all = action

        self.set_joint_torques(action_all)
        self._rate.sleep()

        obs = self.get_observation()
        reward, end_episode_immediately, env_info = self._compute_reward(obs, action)
        return obs, reward, end_episode_immediately, env_info


    """
    Should contain task dependent logic for computing rewards, deciding whether any
    termination conditions have been met, and any other information to be logged.

    Should be overwritten by subclasses.
    """
    def _compute_reward(self, obs, action):
        end_episode_immediately = False
        reward = 0
        env_info = dict(
            actual_torques=self.get_joint_torques()
        )
        return reward, end_episode_immediately, env_info


    """
    Returns the Cartesian coordinates of hte end effector.
    """
    def get_end_effector_pos(self):
        state_dict = self._arm.endpoint_pose()
        pos = state_dict['position']
        return np.array([
            pos.x,
            pos.y,
            pos.z
        ])

    """
    Proportional-Derivative controller used to set joint angles to a given position.
    """
    def _get_pd_torques(self, error, p_coeff=15, d_coeff=5):
        joint_velocities = self.get_joint_velocities()

        torques = - p_coeff*error - d_coeff*joint_velocities
        return torques

    """
    Uses the PD controller to move the arm into the NEUTRAL joint position.
    """
    def reset_arm_to_neutral(self):
        r = rospy.Rate(10)
        for i in range(100):
            joint_angles = self.get_joint_angles()
            error = joint_angles - NEUTRAL
            torques = self._get_pd_torques(error, 30, 5)
            self.set_joint_torques(torques)

            """
            Terminates the procedure once the arm is sufficiently close to the specified
            joint angles 
            """
            if sum(error ** 2) < 0.05:
                break
            r.sleep()


    def get_observation(self):
        pos = self.get_joint_angles()[self._joint_mask]
        vel = self.get_joint_velocities()[self._joint_mask]
        obs = np.concatenate((pos, vel))
        if self._include_xpos:
            obs = np.concatenate([obs, self.get_end_effector_pos()])
        if self._include_pose:
            obs = np.concatenate([obs, self.compute_euler_angles()])
        if self._include_actual_torques:
            obs = np.concatenate([obs, self.get_joint_torques()])
        return obs

    def compute_euler_angles(self):
        quat = self._arm.endpoint_pose()['orientation']
        return tf.transformations.euler_from_quaternion(quat)

    def clean_shutdown(self):
        print("\nClosing rllab interface.")
        self._arm.exit_control_mode()

    def map_values_to_joints(self, values):
        """
        limb: String 'left' or 'right'
        values: List of numerical values to map to joints using above ordering
                Order '_s0','_s1','_e0','_e1','_w0','_w1','_w2'
        -----
        Returns a dictionary with joint_name:value
        """
        joint_names = self._arm.joint_names()
        return dict(zip(joint_names, values))

    def get_joint_angles(self):
        """
        Returns list of joint angles
        """
        ja_dict = self._arm.joint_angles()
        return np.array([ja_dict[k] for k in sorted(ja_dict)])

    def get_joint_velocities(self):
        """
        Returns list of joint velocities
        """
        jv_dict = self._arm.joint_velocities()
        return np.array([jv_dict[k] for k in sorted(jv_dict)])

    def get_joint_torques(self):
        """
        Returns list of joint torques.
        """
        jt_dict = self._arm.joint_efforts()
        return np.array([jt_dict[k] for k in sorted(jt_dict)])

    def set_joint_torques(self, torques):
        """
        """
        if np.isnan(torques).any() or len(torques) != 7:
            return

        torques = np.clip(np.asarray(torques),
                          -self.MAX_TORQUES, self.MAX_TORQUES)

        torques_dict = self.map_values_to_joints(torques)
        self._arm.set_joint_torques(torques_dict)


    """
    Zeros out torques sent to the robot, allowing one to freely manipulate the arm by hand.
    The arm should remain stationary while this is active, but in practice, gravity compensation
    may cause the arm to rise gently instead.
    """
    def float(self, time):
        r = rospy.Rate(10)
        for i in range(int(time * 10)):
           torques = np.zeros(self._Da)
           self.set_joint_torques(torques)
           r.sleep()


    """Uses intera's own controller to move joints to specified joint angles"""
    def set_joint_angles(self, joint_angles):
        angle_map = self.map_values_to_joints(joint_angles)
        self._arm.move_to_joint_positions(angle_map)

    def start_observing_joints(self):
        r = rospy.Rate(10)
        while True:
            print(self.get_observation())
            r.sleep()

    def print_logging_observations(self):
        print("Joint Angles", self.get_joint_angles())
        print("Joint Velocities", self.get_joint_velocities())
        print("Joint Cartesian Coordinates", self.get_cartesian_positions())


if __name__ == '__main__':
    env = SawyerEnv()
    env.initialize()
    import IPython
    IPython.embed()