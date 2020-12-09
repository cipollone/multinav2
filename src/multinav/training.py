"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy

from multinav.envs.ros_controls import RosGoalEnv


def train_on_ros():
    """Train an agent on ROS.

    This function is really experimental. It's just used to produce the first
    results.
    """
    # Make env
    env = RosGoalEnv()

    # Start library
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("data/deeq_ros_agent")
