"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

import os

from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.deepq.policies import MlpPolicy

from multinav.envs.ros_controls import RosGoalEnv
from multinav.helpers.general import QuitWithResources
from multinav.helpers.misc import prepare_directories


def train_on_ros():
    """Train an agent on ROS.

    This function is really experimental. It's just used to produce the first
    results.
    """
    # Init for outputs
    model_path, log_path = prepare_directories("ros-stage")

    # Make env
    env = RosGoalEnv(time_limit=100)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=model_path, name_prefix="model"
    )

    # Define agent
    model = DQN(
        policy=MlpPolicy,
        env=env,
        double_q=True,
        prioritized_replay=True,
        learning_starts=5000,
        tensorboard_log=log_path,
        verbose=1,
    )

    # Behaviour on quit
    QuitWithResources.add(
        "last_save", lambda: model.save(os.path.join(model_path, "model"))
    )

    # Start
    model.learn(
        total_timesteps=2000000,
        callback=checkpoint_callback,
    )

    # Save weights
    model.save(os.path.join(model_path, "model"))
