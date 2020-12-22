"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

import json
import os

from stable_baselines import DQN
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.deepq.policies import LnMlpPolicy

from multinav.envs.cont_sapientino import make_sapientino_cont_env
from multinav.envs.ros_controls import make_ros_env
from multinav.helpers.general import QuitWithResources
from multinav.helpers.misc import prepare_directories
from multinav.helpers.stable_baselines import CustomCheckpointCallback, RendererCallback

# Default environments and algorithms parameters
#   Always prefer to specify them with a json; do not rely on defaults.
default_parameters = dict(
    episode_time_limit=100,
    save_freq=1000,
    learning_starts=5000,
    total_timesteps=2000000,
    resume_file=None,
    log_interval=100,  # In #of episodes
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    # Ros parameters section
    notmoving_limit=12,
    gamma=0.99,
    learning_rate=5e-4,
    acceleration=0.02,
    # Sapientino parameters section
    angular_acceleration=20.0,
    max_velocity=0.20,
    min_velocity=0.0,
    max_angular_vel=40,
    initial_position=[1, 1],
)


def train(env_name, json_params=None):
    """Train an agent on the ROS environment.

    :param env_name: the environment id (see ``multinav --help``)
    :param json_params: the path (str) of json file of parameters.
    """
    # Settings
    params = dict(default_parameters)
    if json_params:
        with open(json_params) as f:
            loaded_params = json.load(f)
        params.update(loaded_params)

    # Init output directories and save params
    resuming = bool(params["resume_file"])
    model_path, log_path = prepare_directories(
        env_name=env_name,
        resuming=resuming,
        args=params,
    )

    # Make environment
    if env_name == "ros":
        env = make_ros_env(params=params)
    else:
        make_sapientino_cont_env(params=params)  # Just for vulture; still not supported
        raise RuntimeError("Environment not supported")

    # Callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path,
        save_freq=params["save_freq"],
        extra=None,
    )
    renderer_callback = RendererCallback()
    all_callbacks = CallbackList([renderer_callback, checkpoint_callback])

    # Define agent
    if not resuming:
        model = DQN(
            policy=LnMlpPolicy,
            env=env,
            gamma=params["gamma"],
            learning_rate=params["learning_rate"],
            double_q=True,
            learning_starts=params["learning_starts"],
            prioritized_replay=True,
            exploration_fraction=params["exploration_fraction"],
            exploration_final_eps=params["exploration_final_eps"],
            exploration_initial_eps=params["exploration_initial_eps"],
            tensorboard_log=log_path,
            full_tensorboard_log=False,
            verbose=1,
        )
    else:
        # TODO: continue here
        # Reload model
        model, _, counters = checkpoint_callback.load(
            path=params["resume_file"],
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
        total_timesteps=params["total_timesteps"],
        log_interval=params["log_interval"],
        callback=all_callbacks,
        reset_num_timesteps=not resuming,
    )

    # Save weights
    model.save(os.path.join(model_path, "model"))
