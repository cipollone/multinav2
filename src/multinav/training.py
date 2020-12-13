"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

import json
import os

from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.deepq.policies import MlpPolicy

from multinav.envs.ros_controls import RosGoalEnv
from multinav.helpers.general import QuitWithResources
from multinav.helpers.misc import prepare_directories


# TODO: resume with args
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
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path, save_freq=1000, name_prefix="dqn"
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
        "last_save", lambda: checkpoint_callback.save(
            step=checkpoint_callback.num_timesteps)
    )

    # Start
    model.learn(
        total_timesteps=2000000,
        callback=checkpoint_callback,
    )

    # Save weights
    model.save(os.path.join(model_path, "model"))


class CustomCheckpointCallback(BaseCallback):
    """Manage model checkpoints.

    This class manages checkpoints, save and restore.
    It also act as callback class so that it can be used inside
    stable_baselines learning loop.
    If you don't plan to use it as callback, assign the model to self.model.
    """

    def __init__(self, save_path, save_freq=None, name_prefix="model"):
        """Initialize.

        :param save_path: model checkpoints path.
        :param save_freq: number of steps between each save (None means never).
        :param name_prefix: just a name for the saved weights.
        """
        BaseCallback.__init__(self)

        # Store
        self._save_freq = save_freq
        self._counters_file = os.path.join(
            save_path, os.path.pardir, "counters.json")
        self._chkpt_format = os.path.join(
            save_path, name_prefix + "_{step}")
        self._chkpt_extension = ".zip"

    def _update_counters(self, filepath, step):
        """Update the file of counters with a new entry.

        :param filepath: checkpoint that is being saved
        :param step: current global step
        """
        counters = {}

        # Load
        if os.path.exists(self._counters_file):
            with open(self._counters_file) as f:
                counters = json.load(f)

        counters[filepath] = dict(step=step)

        # Save
        with open(self._counters_file, "w") as f:
            json.dump(counters, f, indent=4)

    def save(self, step):
        """Manually save a checkpoint.

        :param step: the current step of the training
            (used just to identify checkpoints).
        """
        filepath = self._chkpt_format.format(step=step)
        self.model.save(filepath)
        self._update_counters(
            filepath=filepath + self._chkpt_extension, step=step)

    def load(self, path, env):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path.
        :param env: the environment.
        :return: the model and associated timestep.
        """
        # Restore
        model = DQN.load(load_path=path, env=env)
        print("> Loaded:", path)

        # Read counters
        with open(self._counters_file) as f:
            data = json.load(f)
        step = data[path]

        return model, step

    def _on_step(self):
        """Authomatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)
