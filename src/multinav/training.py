"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

from gym.wrappers import TimeLimit
import json
import os
from pathlib import Path
import pickle

from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import SapientinoConfiguration, SapientinoAgentConfiguration

from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback, CallbackList
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.deepq.policies import LnMlpPolicy

from multinav.algorithms.modular_dqn import LnMlpModularPolicy
from multinav.envs.ros_controls import RosControlsEnv, RosGoalEnv, RosTerminationEnv
from multinav.helpers.general import QuitWithResources
from multinav.helpers.misc import prepare_directories
from multinav.restraining_bolts.rb_grid_sapientino import GridSapientinoRB
from multinav.wrappers.sapientino import GridRobotFeatures, ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper


def train(env, json_args=None):
    """Train an agent on the ROS environment.

    :param env: the environment id {"ros", "sapientino-cont"}
    :param json_args: the path (str) of json file of arguments.
    """
    # Defaults. Please use json_args. These might be good just for ros
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
        angular_speed=10.0,      # Sapientino only here and below
        max_velocity=20.0,
        min_velocity=0.0,
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
    if env == "ros":
        input_env = make_ros_env(learning_params)
    elif env == "sapientino-cont":
        input_env = make_sapientino_cont_env(learning_params)
    else:
        raise RuntimeError("not a valid environment name")

    # Normalize the features
    # TODO: reenable, but before temporal goal!
    #venv = DummyVecEnv([lambda: input_env])
    #norm_env = VecNormalize(
    #    venv=venv,
    #    norm_obs=True,
    #    norm_reward=False,
    #    gamma=learning_params["gamma"],
    #    training=True,
    #)

    # Callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path,
        normalizer=None,   # TODO
        save_freq=learning_params["save_freq"],
        name_prefix="dqn",
    )
    renderer_callback = RendererCallback()
    all_callbacks = CallbackList([renderer_callback, checkpoint_callback])

    # Define agent
    if not resuming:
        model = DQN(
            policy=LnMlpPolicy, # TODO: use the modified model
            env=input_env,      # TODO
            gamma=learning_params["gamma"],
            learning_rate=learning_params["learning_rate"],
            double_q=True,
            learning_starts=learning_params["learning_starts"],
            prioritized_replay=True,
            exploration_fraction=learning_params["exploration_fraction"],
            exploration_final_eps=learning_params["exploration_final_eps"],
            exploration_initial_eps=learning_params["exploration_initial_eps"],
            tensorboard_log=log_path,
            full_tensorboard_log=False,
            verbose=1,
        )
    else:
        # Reload model
        model, norm_env, counters = checkpoint_callback.load(
            path=learning_params["resume_file"],
        )
        # TODO
        # Reapply normalizer to env
        # norm_env.set_venv(venv)
        # model.set_env(norm_env)
        # Restore counters
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
        callback=all_callbacks,
        reset_num_timesteps=not resuming,
    )

    # Save weights
    model.save(os.path.join(model_path, "model"))


def make_ros_env(learning_params):
    """Return the ros environment."""
    input_env = RosGoalEnv(
        env=RosTerminationEnv(
            env=RosControlsEnv(),
            time_limit=learning_params["episode_time_limit"],
            notmoving_limit=learning_params["notmoving_limit"],
        )
    )
    return input_env


def make_sapientino_cont_env(learning_params):
    """Return sapientino continuous state environment."""
    nb_colors = 3
    agent_configuration = SapientinoAgentConfiguration(continuous=True)
    configuration = SapientinoConfiguration(
        [agent_configuration],
        path_to_map=Path("small-sapientino-map.txt"),
        reward_per_step=-0.01,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        angular_speed=learning_params["angular_speed"],
        max_velocity=learning_params["max_velocity"],
        min_velocity=learning_params["min_velocity"],
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))
    tg = GridSapientinoRB(nb_colors).make_sapientino_goal()
    env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
    env = TimeLimit(env, max_episode_steps=learning_params["episode_time_limit"])
    print("Temporal goal:", tg._formula)

    return env


class CustomCheckpointCallback(BaseCallback):
    """Manage model checkpoints.

    This class manages checkpoints, save and restore.
    It also act as callback class so that it can be used inside
    stable_baselines learning loop.
    If you don't plan to use it as callback, assign the model to self.model.
    """

    def __init__(self, save_path, normalizer, save_freq=None, name_prefix="model"):
        """Initialize.

        :param save_path: model checkpoints path.
        :param normalizer: a VecNormalize instance.
        :param save_freq: number of steps between each save (None means never).
        :param name_prefix: just a name for the saved weights.
        """
        BaseCallback.__init__(self)

        # Store
        self.normalizer_model = normalizer
        self._save_freq = save_freq
        self._counters_file = os.path.join(save_path, os.path.pardir, "counters.json")
        self._chkpt_format = os.path.join(save_path, name_prefix + "_{step}")
        self._chkpt_extension = ".zip"
        self._normalizer_format = os.path.join(save_path, "VecNormalize_{step}.pickle")

    def _update_counters(self, filepath, step, normalizer_file):
        """Update the file of counters with a new entry.

        :param filepath: checkpoint that is being saved
        :param step: current global step
        :param normalizer_file: associated normalizer
        """
        counters = {}

        # Load
        if os.path.exists(self._counters_file):
            with open(self._counters_file) as f:
                counters = json.load(f)

        filepath = os.path.relpath(filepath)
        counters[filepath] = dict(step=step, normalizer=normalizer_file)

        # Save
        with open(self._counters_file, "w") as f:
            json.dump(counters, f, indent=4)

    def save(self, step):
        """Manually save a checkpoint.

        :param step: the current step of the training
            (used just to identify checkpoints).
        """
        # Save model
        model_path = self._chkpt_format.format(step=step)
        self.model.save(model_path)
        # Save checkpoint
        normalizer_path = self._normalizer_format.format(step=step)
        with open(normalizer_path, "wb") as f:
            pickle.dump(self.normalizer_model, f)

        self._update_counters(
            filepath=model_path + self._chkpt_extension,
            step=step,
            normalizer_file=normalizer_path,
        )

    def load(self, path):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path.
        :return: the model and associated counters.
        """
        # Restore
        path = os.path.relpath(path)
        model = DQN.load(load_path=path)
        print("> Loaded:", path)

        # Read counters
        with open(self._counters_file) as f:
            data = json.load(f)
        counters = data[path]

        # Restore normalizer
        normalizer_path = counters.pop("normalizer")
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)

        return model, normalizer, counters

    def _on_step(self):
        """Automatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)


class RendererCallback(BaseCallback):
    """Just render at each frame."""

    def _on_step(self):
        """Do it."""
        self.training_env.render()
