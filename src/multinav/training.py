"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

import json
import os
from pathlib import Path

from flloat.semantics import PLInterpretation
from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from stable_baselines import DQN
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.deepq.policies import LnMlpPolicy

from multinav.envs.ros_controls import RosControlsEnv, RosGoalEnv, RosTerminationEnv
from multinav.helpers.general import QuitWithResources
from multinav.helpers.misc import prepare_directories
from multinav.helpers.stable_baselines import CustomCheckpointCallback, RendererCallback
from multinav.restraining_bolts.automata import make_sapientino_goal_with_automata
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper

# TODO: move features extraction in a different module


def train(env_name, json_args=None):
    """Train an agent on the ROS environment.

    :param env_name: the environment id {"ros", "sapientino-cont"}
    :param json_args: the path (str) of json file of arguments.
    """
    # Defaults. Please use json_args, this dict just show the supported fields
    #  but they must be tuned
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
        notmoving_limit=12,  # Ros only
        gamma=0.99,
        learning_rate=5e-4,
        acceleration=0.02,  # Sapientino only here and below
        angular_acceleration=20.0,
        max_velocity=0.20,
        min_velocity=0.0,
        max_angular_vel=40,
        initial_position=[1, 1],
    )
    # Settings
    if json_args:
        with open(json_args) as f:
            params = json.load(f)
        learning_params.update(params)

    # Init for outputs
    resuming = bool(learning_params["resume_file"])
    model_path, log_path = prepare_directories(
        env_name=env_name,
        resuming=resuming,
        args=learning_params,
    )

    # Make env
    if env_name == "ros":
        input_env = make_ros_env(learning_params)
    elif env_name == "sapientino-cont":
        input_env = make_sapientino_cont_env(learning_params)
    else:
        raise RuntimeError("not a valid environment name")

    # Normalize the features
    # TODO: reenable, but before temporal goal!
    # venv = DummyVecEnv([lambda: input_env])   # noqa: E800  (becaue there's
    # env = VecNormalize(                       # noqa: E800   the todo)
    #    venv=venv,                             # noqa: E800
    #    norm_obs=True,                         # noqa: E800
    #    norm_reward=False,                     # noqa: E800
    #    gamma=learning_params["gamma"],        # noqa: E800
    #    training=True,                         # noqa: E800
    # )                                         # noqa: E800
    env = input_env

    # Callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path,
        normalizer=None,  # TODO: readd normalizer?
        save_freq=learning_params["save_freq"],
        name_prefix="dqn",
    )
    renderer_callback = RendererCallback()
    all_callbacks = CallbackList([renderer_callback, checkpoint_callback])

    # Define agent
    if not resuming:
        model = DQN(
            policy=LnMlpPolicy,  # TODO: use the rb model
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
            full_tensorboard_log=False,
            verbose=1,
        )
    else:
        # Reload model
        model, _, counters = checkpoint_callback.load(
            path=learning_params["resume_file"],
        )
        # TODO
        # Reapply normalizer to env
        # norm_env.set_venv(venv)  # noqa: E800  (becaue there's the todo)
        # model.set_env(norm_env)  # noqa: E800
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
    # Define the robot
    agent_configuration = SapientinoAgentConfiguration(
        continuous=True,
        initial_position=learning_params["initial_position"],
    )
    # Define the environment
    configuration = SapientinoConfiguration(
        [agent_configuration],
        path_to_map=Path("inputs/sapientino-map.txt"),
        reward_per_step=-0.01,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        acceleration=learning_params["acceleration"],
        angular_acceleration=learning_params["angular_acceleration"],
        max_velocity=learning_params["max_velocity"],
        min_velocity=learning_params["min_velocity"],
        max_angular_vel=learning_params["angular_acceleration"],
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    colors = ["red", "green", "blue"]

    def extract_sapientino_fluents(obs, action):
        """Extract Sapientino fluents."""
        is_beep = obs.get("beep") > 0
        color_id = obs.get("color")
        if is_beep and 0 <= color_id - 1 < len(colors):
            color = colors[color_id - 1]
            fluents = {color} if color in colors else set()
        else:
            fluents = set()
        return PLInterpretation(fluents)

    # Define the temporal goal
    tg = make_sapientino_goal_with_automata(colors, extract_sapientino_fluents)
    env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
    env = TimeLimit(env, max_episode_steps=learning_params["episode_time_limit"])
    print("Temporal goal:", tg._formula)

    return env
