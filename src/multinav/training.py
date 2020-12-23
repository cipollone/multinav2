"""This module implements the general logic of the training loop.

This file is likely to change. I just needed an outer module where to put the
general logic.
"""

import json

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
    # Common
    resume_file=None,
    episode_time_limit=100,
    # DQN params
    gamma=0.99,
    learning_rate=5e-4,
    learning_starts=5000,
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    save_freq=1000,
    log_interval=100,  # In #of episodes
    total_timesteps=2000000,
    # Ros agent env
    notmoving_limit=12,
    # Sapientino env
    acceleration=0.02,
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
        trainer = TrainStableBaselines(
            env=make_ros_env(params=params),
            params=params,
            model_path=model_path,
            log_path=log_path,
        )
    elif env_name == "sapientino-cont":
        trainer = TrainStableBaselines(
            env=make_sapientino_cont_env(params=params),
            params=params,
            model_path=model_path,
            log_path=log_path,
        )
    else:
        raise RuntimeError("Environment not supported")

    # Start
    trainer.train()


class TrainStableBaselines:
    """Define the agnent and training loop for stable_baselines."""

    def __init__(self, env, params, model_path, log_path):
        """Initialize.

        :param env: gym environment.
        :param params: dict of parameters, like `default_parameters`.
        :param model_path: directory where to save models.
        :param log_path: directory where to save tensorboard logs.
        """
        # Callbacks
        checkpoint_callback = CustomCheckpointCallback(
            save_path=model_path,
            save_freq=params["save_freq"],
            extra=None,
        )
        renderer_callback = RendererCallback()
        all_callbacks = CallbackList([renderer_callback, checkpoint_callback])

        # Define agent
        resuming = bool(params["resume_file"])
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
            # Reload model
            model, _, counters = checkpoint_callback.load(
                path=params["resume_file"],
            )
            # Restore properties
            model.tensorboard_log = log_path
            model.num_timesteps = counters["step"]
            model.set_env(env)

        # Store
        self.params = params
        self.resuming = resuming
        self.saver = checkpoint_callback
        self.callbacks = all_callbacks
        self.model = model

    def train(self):
        """Do train.

        Interrupt at any type with Ctrl-C.
        """
        # Behaviour on quit
        QuitWithResources.add(
            "last_save",
            lambda: self.saver.save(step=self.saver.num_timesteps),
        )

        # Start
        self.model.learn(
            total_timesteps=self.params["total_timesteps"],
            log_interval=self.params["log_interval"],
            callback=self.callbacks,
            reset_num_timesteps=not self.resuming,
        )

        # Final save
        self.saver.save(self.params["total_timesteps"])
