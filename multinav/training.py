# -*- coding: utf-8 -*-
#
# Copyright 2020 Roberto Cipollone, Marco Favorito
#
# ------------------------------
#
# This file is part of multinav.
#
# multinav is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multinav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multinav.  If not, see <https://www.gnu.org/licenses/>.
#
"""This module implements the general logic of the training loop."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np
from gym import Env
from matplotlib import pyplot as plt
from PIL import Image
from stable_baselines import DQN
from stable_baselines.common.callbacks import CallbackList

from multinav.algorithms.agents import QFunctionModel, ValueFunctionModel
from multinav.algorithms.modular_dqn import ModularPolicy
from multinav.algorithms.q_learning import QLearning
from multinav.algorithms.value_iteration import pretty_print_v, value_iteration
from multinav.envs import (
    env_abstract_sapientino,
    env_cont_sapientino,
    env_grid_sapientino,
    env_ros_controls,
)
from multinav.helpers.callbacks import CallbackList as CustomCallbackList
from multinav.helpers.callbacks import FnCallback, SaverCallback
from multinav.helpers.general import QuitWithResources
from multinav.helpers.gym import find_wrapper
from multinav.helpers.misc import prepare_directories
from multinav.helpers.stable_baselines import (
    CustomCheckpointCallback,
    RendererCallback,
    StatsLoggerCallback,
)
from multinav.wrappers.reward_shaping import (
    RewardShapingWrapper,
    UnshapedEnv,
    UnshapedEnvWrapper,
)
from multinav.wrappers.temprl import BoxAutomataStates
from multinav.wrappers.training import NormalizeEnvWrapper
from multinav.wrappers.utils import CallbackWrapper, MyStatsRecorder, Renderer

# Default environments and algorithms parameters
#   Always prefer to specify them with a json; do not rely on defaults.
default_parameters = dict(
    # Common
    resume_file=None,
    initialize_file=None,
    shaping=None,
    shaping_passive=False,
    active_passive_agents=False,
    dfa_shaping=False,
    action_bias=None,
    action_bias_eps=0.0,
    episode_time_limit=100,
    learning_rate=5e-4,
    gamma=0.99,
    end_on_failure=False,
    log_interval=100,  # In #of episodes
    render=False,
    test_passive=False,
    tg_automaton="inputs/automaton.pickle",
    run_id=None,
    # DQN params
    batch_size=32,
    layers=[64, 64],
    shared_layers=1,
    dueling=False,
    layer_norm=True,
    learning_starts=5000,
    train_freq=2,
    target_network_update_freq=10000,
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    save_freq=100000,
    total_timesteps=2000000,
    buffer_size=50000,
    # Q params
    #   total_timesteps
    nb_episodes=1000,
    q_eps=0.5,
    learning_rate_end=0.0,
    epsilon_end=0.05,
    # ValueIteration params
    max_iterations=2000,
    # Ros agent env
    notmoving_limit=12,
    # Sapientino env
    acceleration=0.02,
    angular_acceleration=20.0,
    max_velocity=0.20,
    min_velocity=0.0,
    max_angular_vel=40,
    initial_position=[1, 1],
    tg_reward=1.0,
    reward_per_step=-0.01,
    reward_outside_grid=0.0,
    reward_duplicate_beep=-0.5,
    # Abs sapientino env
    nb_rooms=2,
    nb_colors=3,
    sapientino_fail_p=0.2,
)


def train(
    env_name: str,
    json_params: Optional[str] = None,
    cmd_params: Optional[Dict[str, Any]] = None,
):
    """Train an agent on the ROS environment.

    :param env_name: the environment id (see ``multinav --help``)
    :param json_params: the path (str) of json file of parameters.
    :param cmd_params: optional command line parameters. These are meant to
        override json_params.
    """
    # Settings
    params = dict(default_parameters)
    if json_params:
        with open(json_params) as f:
            loaded_params = json.load(f)
        params.update(loaded_params)
    if cmd_params:
        params.update(cmd_params)

    # Init output directories and save params
    resuming = any(
        [params["resume_file"], params["initialize_file"], params["action_bias"]]
    )
    model_path, log_path = prepare_directories(
        env_name=env_name,
        resuming=resuming,
        args=params,
        run_id=params["run_id"],
    )

    # Make
    trainer: Trainer
    if env_name == "ros":
        # Ros env
        env = env_ros_controls.make(params=params)

        # Trainer
        trainer = TrainStableBaselines(
            env=env_ros_controls.make(params=params),
            params=params,
            model_path=model_path,
            log_path=log_path,
        )
        # TODO: define _load_cont_shaping_agent just like load_grid_shaping_agent

    elif env_name == "sapientino-cont":
        # Continuous env
        shaping_agent = load_grid_shaping_agent(params)
        env = env_cont_sapientino.make(
            params=params,
            log_dir=log_path,
            shaping_agent=shaping_agent,
        )
        if params["render"]:
            env = Renderer(env)

        # Trainer
        trainer = TrainStableBaselines(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        )

    elif env_name == "sapientino-grid":
        # Grid env
        env = env_grid_sapientino.make(params=params, log_dir=log_path)
        if params["render"]:
            env = Renderer(env)

        # Trainer
        trainer = TrainQ(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        )

    elif env_name == "sapientino-abs":
        # Abstract env
        env = env_abstract_sapientino.make(
            params=params,
            log_dir=log_path,
        )
        # Trainer
        trainer = TrainValueIteration(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        )
    else:
        raise RuntimeError("Environment not supported")

    # Start
    trainer.train()


class Trainer(ABC):
    """Trainer interface (used for typing)."""

    @abstractmethod
    def train(self) -> None:
        """Trainig loop."""
        pass


class TrainStableBaselines(Trainer):
    """Define the agnent and training loop for stable_baselines."""

    def __init__(self, env: Env, params: dict, model_path: str, log_path: str):
        """Initialize.

        :param env: gym environment. Assuming observation space is a tuple,
            where first component is from original env, and the second is
            temporal goal state.
        :param params: dict of parameters, like `default_parameters`.
        :param model_path: directory where to save models.
        :param log_path: directory where to save tensorboard logs.
        """
        # Check
        if params["initialize_file"]:
            raise ValueError("Initialization not supported; use resuming option")
        if params["action_bias"]:
            raise ValueError("Action bias is not maintained here")

        # Alias
        original_env = env

        # Load a saved agent for the action bias
        self.biased_agent: Optional[DQN] = None
        if params["action_bias"]:
            loading_params = dict(params)
            loading_params["resume_file"] = params["action_bias"]
            loading_params["action_bias"] = None

            self.biased_agent = TrainStableBaselines(
                env=env,
                params=loading_params,
                model_path=model_path,
                log_path=log_path,
            ).model

        # Collect statistics
        #    (assuming future wrappers do not modify episodes)
        env = MyStatsRecorder(env=env, gamma=params["gamma"])

        # Callbacks
        checkpoint_callback = CustomCheckpointCallback(
            save_path=model_path,
            save_freq=params["save_freq"],
            extra=None,
        )
        stats_logger_callback = StatsLoggerCallback(stats_recorder=env, scope="env0")

        callbacks_list = [checkpoint_callback, stats_logger_callback]
        if params["render"]:
            renderer_callback = RendererCallback()
            callbacks_list.append(renderer_callback)

        # If training a passive agent log this too
        if params["active_passive_agents"]:

            # Find the reward shaping env
            reward_shaping_env = find_wrapper(env, RewardShapingWrapper)

            passive_stats_env = MyStatsRecorder(
                env=UnshapedEnv(reward_shaping_env),
                gamma=params["gamma"],
            )

            passive_stats_callback = StatsLoggerCallback(
                stats_recorder=passive_stats_env,
                scope="env1",
            )
            callbacks_list.append(passive_stats_callback)

            # Make it move with the original env
            env = UnshapedEnvWrapper(
                shaped_env=env,
                unshaped_env=passive_stats_env,
            )
            original_reward_getter = env.get_reward  # alias
        else:
            original_reward_getter = None

        # Combine callbacks
        all_callbacks = CallbackList(callbacks_list)

        # Define or load
        resuming = bool(params["resume_file"])
        if not resuming:
            # Normalizer
            normalized_env = NormalizeEnvWrapper(
                env=env,
                training=True,
                entry=0,  # Only env features, not temporal goal state
            )
            flat_env = BoxAutomataStates(normalized_env)
            # Saving normalizer too
            checkpoint_callback.saver.extra_model = normalized_env

            # Agent
            model = DQN(
                env=flat_env,
                policy=ModularPolicy,
                policy_kwargs={
                    "layer_norm": params["layer_norm"],
                    "layers": params["layers"],
                    "shared_layers": params["shared_layers"],
                    "dueling": params["dueling"],
                },
                gamma=params["gamma"],
                learning_rate=params["learning_rate"],
                train_freq=params["train_freq"],
                double_q=True,
                batch_size=params["batch_size"],
                buffer_size=params["buffer_size"],
                learning_starts=params["learning_starts"],
                prioritized_replay=True,
                target_network_update_freq=params["target_network_update_freq"],
                exploration_fraction=params["exploration_fraction"],
                exploration_final_eps=params["exploration_final_eps"],
                exploration_initial_eps=params["exploration_initial_eps"],
                active_passive_agents=params["active_passive_agents"],
                passive_reward_getter=original_reward_getter,
                tensorboard_log=log_path,
                full_tensorboard_log=False,
                verbose=1,
            )
        else:
            # Reload model
            model, extra_model, counters = checkpoint_callback.load(
                path=params["resume_file"],
            )

            # Restore normalizer and env
            normalized_env = extra_model
            normalized_env.set_env(env)
            flat_env = BoxAutomataStates(normalized_env)

            # Restore properties
            model.tensorboard_log = log_path
            model.num_timesteps = counters["step"]
            model.learning_starts = params["learning_starts"] + counters["step"]
            model.set_env(flat_env)
            model.passive_reward_getter = original_reward_getter

        # Store
        self.params = params
        self.resuming = resuming
        self.saver = checkpoint_callback
        self.logger = stats_logger_callback
        self.callbacks = all_callbacks
        self.model: DQN = model
        self.normalized_env = normalized_env
        self.testing_agent = model if not params["test_passive"] else model.passive_agent

    def train(self):
        """Do train.

        Interrupt at any type with Ctrl-C.
        """
        # Behaviour on quit
        QuitWithResources.add(
            "last_save",
            lambda: self.saver.save(step=self.saver.num_timesteps),
        )
        QuitWithResources.add("last_log", self.logger.log)

        # Start
        self.model.learn(
            total_timesteps=self.params["total_timesteps"],
            log_interval=self.params["log_interval"],
            callback=self.callbacks,
            reset_num_timesteps=not self.resuming,
        )

        # Final save
        self.saver.save(self.params["total_timesteps"])


class TrainQ(Trainer):
    """Agent and training loop for Q learning."""

    def __init__(
        self,
        env: Optional[Env],
        params: Dict[str, Any],
        model_path: str,
        log_path: Optional[str],
        agent_only: bool = False,
    ):
        """Initialize.

        :param env: discrete-state gym environment.
        :param params: dict of parameters. See `default_parameters`.
        :param model_path: directory where to save models.
        :param log_path: directory where to save training logs.
        :param agent_only: if True, it defines or loads the agent(s),
            then stops. The environment is not used. For training,
            this must be false.
        """
        # Check
        if "resume_file" in params and params["resume_file"]:
            raise TypeError("Resuming a training is not supported here")

        # Define agent(s)
        self.agent = QFunctionModel(q_function=dict())
        self.passive_agent = QFunctionModel(q_function=dict())
        self._reinitialized = False

        # Saver
        self.saver = SaverCallback(
            save_freq=None,  # Not needed
            saver=self.agent.save,
            loader=self.agent.load,
            save_path=model_path,
            name_prefix="model",
            model_ext=self.agent.file_ext,
            extra=self.passive_agent.q_function,
        )

        # Load agent
        if params["initialize_file"] is not None:
            self.agent, extra, _ = self.saver.load(params["initialize_file"])
            self.passive_agent.q_function = extra
            # Update saver
            self.saver.saver = self.agent.save
            self.saver.loader = self.agent.load
            self.saver.extra_model = extra
            self._reinitialized = True

            # Save an agent that may be used for testing
            self.testing_agent = self.agent if not params["test_passive"] else self.passive_agent

        # Q function used just for action bias
        self.biased_Q = None
        if params["action_bias"]:
            self.biased_Q = QFunctionModel.load(path=params["action_bias"]).q_function

        # Maybe stop
        if agent_only:
            return
        assert log_path is not None  # mypy

        # Store
        self.env = env
        self.params = params
        self._log_path = log_path

        # Logger
        logger = FnCallback(
            ep_freq=params["log_interval"], ep_fn=lambda _o, _i: self.log()
        )

        # Stats recorder
        self.stats_env = MyStatsRecorder(self.env, gamma=params["gamma"])
        self.env = self.stats_env
        stats_envs = [self.stats_env]

        # If training a passive agent log the second environment too
        if params["active_passive_agents"]:

            # Find the reward shaping env
            reward_shaping_env = find_wrapper(self.env, RewardShapingWrapper)

            self.passive_stats_env = MyStatsRecorder(
                env=UnshapedEnv(reward_shaping_env),
                gamma=params["gamma"],
            )
            stats_envs.append(self.passive_stats_env)

            # Make it move with the original env
            self.env = UnshapedEnvWrapper(
                shaped_env=self.env,
                unshaped_env=self.passive_stats_env,
            )
            original_reward_getter: Optional[
                Callable[[], float]
            ] = self.env.get_reward  # alias
        else:
            original_reward_getter = None

        # Wrap callbacks
        self.callbacks = CustomCallbackList([self.saver, logger])
        self.env = CallbackWrapper(env=self.env, callback=self.callbacks)

        # Log properties
        self._log_properties = ["episode_lengths", "episode_returns", "episode_td_max"]
        self._agent_plot_vars: Dict[str, Any] = {}
        self._init_log("agent", self._agent_plot_vars)
        if params["active_passive_agents"]:
            self._passive_agent_plot_vars: Dict[str, Any] = {}
            self._init_log("passive_agent", self._passive_agent_plot_vars)

        # Learner
        self.learner = QLearning(
            env=self.env,
            stats_envs=stats_envs,
            total_timesteps=params["total_timesteps"],
            alpha=params["learning_rate"],
            eps=params["q_eps"],
            gamma=params["gamma"],
            learning_rate_decay=True,
            learning_rate_end=params["learning_rate_end"],
            epsilon_decay=True,
            epsilon_end=params["epsilon_end"],
            action_bias=self.biased_Q,
            action_bias_eps=params["action_bias_eps"],
            initial_Q=self.agent.q_function if self._reinitialized else None,
            active_passive_agents=params["active_passive_agents"],
            passive_reward_getter=original_reward_getter,
            initial_passive_Q=(
                self.passive_agent.q_function if self._reinitialized else None
            ),
            seed=params["seed"],
        )

        # Link trained and saved agents
        self.agent.q_function = self.learner.Q
        self.passive_agent.q_function = self.learner.passive_Q
        self.saver.extra_model = self.passive_agent.q_function

    def _init_log(self, name: str, variables: Dict[str, Any]):
        """Initialize variables related to logging.

        :param name: this name is appended to the saved files.
        :param variables: a namespace of variables initialized by this
            function.
        """
        # Create the plot
        draw_fig, draw_axes = plt.subplots(
            nrows=len(self._log_properties), ncols=1, figsize=(20, 12)
        )
        draw_lines = [None for i in range(len(self._log_properties))]

        # Create log txt file
        log_file = os.path.join(self._log_path, name + "_log.txt")
        self._log_header = ", ".join(self._log_properties) + "\n"
        with open(log_file, "w") as f:
            f.write(self._log_header)

        # Store
        variables["figure"] = draw_fig
        variables["axes"] = draw_axes
        variables["lines"] = draw_lines
        variables["txt_file"] = log_file
        variables["name"] = name

    def train(self):
        """Start training."""
        # Learn
        self.learner.learn()

        # Save
        self.saver.save()

        # Log
        self.log()

    def log(self):
        """Save logs to files and plots."""
        self._log_figure(self._agent_plot_vars, self.stats_env)
        if self.params["active_passive_agents"]:
            self._log_figure(
                variables=self._passive_agent_plot_vars,
                stats_env=self.passive_stats_env,
            )

    def _log_figure(
        self,
        variables: Dict[str, Any],
        stats_env: MyStatsRecorder,
    ):
        """Save logs for one figure.

        :param variables: a namespace of variables for logging. See _init_log.
        :param stats_env: a MyStatsRecorder environment wrapper that
            stores the current statistics.
        """
        for i in range(len(self._log_properties)):
            name = self._log_properties[i]
            ax = variables["axes"][i]
            line = variables["lines"][i]
            data = getattr(stats_env, name)

            if line is None:
                (variables["lines"][i],) = ax.plot(data)
                ax.set_ylabel(name)
                if name == self._log_properties[-1]:
                    ax.set_xlabel("episodes")

            else:
                line.set_data(np.arange(len(data)), data)
                ax.relim()
                ax.autoscale(tight=True)

        variables["figure"].savefig(
            os.path.join(self._log_path, variables["name"] + "_plots.pdf"),
            bbox_inches="tight",
        )

        # Save to file
        n_samples = len(data)
        by_timestep = [
            [getattr(stats_env, name)[i] for name in self._log_properties]
            for i in range(n_samples)
        ]
        lines = [", ".join([str(x) for x in values]) + "\n" for values in by_timestep]
        with open(variables["txt_file"], "w") as f:
            f.write(self._log_header)
            f.writelines(lines)


class TrainValueIteration(Trainer):
    """Agent and training loop for Value Iteration."""

    def __init__(
        self,
        env: Env,
        params: Dict[str, Any],
        model_path: str,
        log_path: str,
    ):
        """Initialize.

        :param env: discrete-state gym environment.
        :param params: dict of parameters. See `default_parameters`.
        :param model_path: directory where to save models.
        :param log_path: directory where to save training logs.
        """
        # Check
        if params["resume_file"]:
            raise TypeError("Resuming a trainingg is not supported for this algorithm.")
        if params["action_bias"]:
            raise TypeError("Action bias is not supported here.")
        if params["active_passive_agents"]:
            raise TypeError("Not training a passive agent here.")

        # Agent
        agent = ValueFunctionModel(
            value_function=dict(),
            policy=dict(),
        )

        # Saver
        self.saver = SaverCallback(
            save_freq=None,  # Not needed
            saver=agent.save,
            loader=agent.load,
            save_path=model_path,
            name_prefix="model",
            model_ext=agent.file_ext,
            extra=None,
        )
        env = CallbackWrapper(env=env, callback=self.saver)

        # Stats recorder
        env = MyStatsRecorder(env, gamma=params["gamma"])

        # Store
        self.env = env
        self.params = params
        self.agent = agent
        self._log_path = log_path
        self.testing_agent = agent

    def train(self):
        """Start training."""
        # Learn
        value_function, policy = value_iteration(
            env=self.env,
            max_iterations=self.params["max_iterations"],
            eps=1e-5,
            discount=self.params["gamma"],
        )
        # Store
        self.agent.value_function = dict(value_function)
        self.agent.policy = dict(policy)

        # Save
        self.saver.save()

        # Log
        self.log()

    def log(self):
        """Log."""
        pretty_print_v(self.agent.value_function)

        print("Policy", self.agent.policy)

        frame = self.env.render(mode="rgb_array")
        img = Image.fromarray(frame)
        img.save(os.path.join(self._log_path, "frame.png"))


def load_grid_shaping_agent(
    params: Dict[str, Any],
) -> Optional[QFunctionModel]:
    """Load a grid-sapientino checkpoint and returns the agent.

    :param params: the parameters of a continuous sapientino.
        (because we'll use this to load an agent for reward shaping).
    :return: a grid sapientino agent or None.
    """
    if not params["shaping"]:
        return None

    # Simple config for grid-sapientino
    grid_params = {}
    grid_params.update(default_parameters)
    grid_params["initialize_file"] = params["shaping"]

    # Load agent from trainer
    trainer = TrainQ(
        env=None,
        params=grid_params,
        model_path=os.path.dirname(params["shaping"]),
        log_path=None,
        agent_only=True,
    )

    # Return the agent
    if params["shaping_passive"]:
        print("> Shaping from passive agent: " + params["shaping"])
        return trainer.passive_agent
    else:
        print("> Shaping from agent: " + params["shaping"])
        return trainer.agent
