"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

import json
import os

from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.deepq.policies import MlpPolicy

from multinav.envs.ros_controls import RosControlsEnv, RosGoalEnv, RosTerminationEnv
from multinav.helpers.general import QuitWithResources
from multinav.helpers.misc import prepare_directories


def train_on_ros(json_args=None):
    """Train an agent on the ROS environment.

    :param json_args: the path (str) of json file of arguments.
    """
    # Defaults
    learning_params = dict(
        episode_time_limit=100,
        save_freq=1000,
        learning_starts=5000,
        total_timesteps=2000000,
        resume_file=None,
        log_interval=100,  # In #of episodes
        exploration_fraction=0.8,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        notmoving_limit=12,
        gamma=0.99,
        learning_rate=5e-4,
    )
    # Settings
    if json_args:
        with open(json_args) as f:
            params = json.load(f)
        learning_params.update(params)

    # Init for outputs
    resuming = bool(learning_params["resume_file"])
    model_path, log_path = prepare_directories(
        env_name="ros-stage",
        resuming=resuming,
        args=learning_params,
    )

    # Make env
    env = RosGoalEnv(
        env=RosTerminationEnv(
            env=RosControlsEnv(),
            time_limit=learning_params["episode_time_limit"],
            notmoving_limit=learning_params["notmoving_limit"],
        )
    )
    # Normalize the features
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        venv=env,
        norm_obs=True,
        norm_reward=False,
        gamma=learning_params["gamma"],
        training=True,
    )

    # Callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path,
        save_freq=learning_params["save_freq"],
        name_prefix="dqn",
    )

    # Define agent
    if not resuming:
        model = DQN(
            policy=MlpPolicy,
            env=env,
            gamma=learning_params["gamma"],
            learning_rate=learning_params["learning_rate"],
            double_q=True,
            learning_starts=learning_params["learning_starts"],
            prioritized_replay=True,
            exploration_fraction=learning_params["exploration_fraction"],
            exploration_final_eps=learning_params["exploration_final_eps"],
            exploration_initial_eps=learning_params["exploration_initial_eps"],
            tensorboard_log=log_path,
            full_tensorboard_log=True,
            verbose=1,
        )
    else:
        model, counters = checkpoint_callback.load(
            path=learning_params["resume_file"],
            env=env,
        )
        model.tensorboard_log = log_path
        model.num_timesteps = counters["step"]

    # Behaviour on quit
    QuitWithResources.add(
        "last_save",
        lambda: checkpoint_callback.save(step=checkpoint_callback.num_timesteps),
    )

    # Start
    model.learn(
        total_timesteps=learning_params["total_timesteps"],
        log_interval=learning_params["log_interval"],
        callback=checkpoint_callback,
        reset_num_timesteps=not resuming,
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
        self._counters_file = os.path.join(save_path, os.path.pardir, "counters.json")
        self._chkpt_format = os.path.join(save_path, name_prefix + "_{step}")
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

        filepath = os.path.relpath(filepath)
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
        self._update_counters(filepath=filepath + self._chkpt_extension, step=step)

    def load(self, path, env):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path.
        :param env: the environment.
        :return: the model and associated timestep.
        """
        # Restore
        path = os.path.relpath(path)
        model = DQN.load(load_path=path, env=env)
        print("> Loaded:", path)

        # Read counters
        with open(self._counters_file) as f:
            data = json.load(f)
        step = data[path]

        return model, step

    def _on_step(self):
        """Automatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)
