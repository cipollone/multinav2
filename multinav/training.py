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

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, cast

import numpy as np
from gym import Env
from matplotlib import pyplot as plt
from PIL import Image

from multinav.algorithms.agents import QFunctionModel, ValueFunctionModel
from multinav.algorithms.q_learning import QLearning
from multinav.algorithms.value_iteration import pretty_print_v, value_iteration
from multinav.envs import env_abstract_sapientino, env_grid_sapientino
from multinav.helpers.callbacks import CallbackList as CustomCallbackList
from multinav.helpers.callbacks import FnCallback, SaverCallback
from multinav.helpers.gym import find_wrapper
from multinav.wrappers.reward_shaping import (
    RewardShapingWrapper,
    UnshapedEnv,
    UnshapedEnvWrapper,
)
from multinav.wrappers.utils import CallbackWrapper, MyStatsRecorder, Renderer


def train(params: Dict[str, Any]):
    """Train an agent.

    :param params: for examples on these parameters see the project repository
        in files under inputs/
    """
    # Get options
    alg_params = params["algorithm"]["params"]
    env_params = params["environment"]["params"]
    env_params["seed"] = params["seed"]
    alg_params["seed"] = params["seed"]
    env_params["gamma"] = alg_params["gamma"]
    env_name = env_params.pop("env")

    # Make
    trainer: Trainer
    if env_name == "level0":
        # Abstract env
        env = env_abstract_sapientino.make(
            params=env_params,
            log_dir=params["logs-dir"],
        )

        # Trainer
        trainer = TrainQ(
            env=env,
            params=alg_params,
            model_path=params["model-dir"],
            log_path=params["logs-dir"],
        )
    elif env_name == "level1":
        # Grid env
        env = env_grid_sapientino.make(
            params=env_params,
            log_dir=params["logs-dir"],
        )
        if env_params["render"]:
            env = Renderer(env)

        # Trainer
        trainer = TrainQ(
            env=env,
            params=alg_params,
            model_path=params["model-dir"],
            log_path=params["logs-dir"],
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


class TrainQ(Trainer):
    """Agent and training loop for Q learning."""

    def __init__(
        self,
        env: Env,
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
            assert isinstance(reward_shaping_env, RewardShapingWrapper), (
                "Expecting a reward shaped Env")

            self.passive_stats_env = MyStatsRecorder(
                env=UnshapedEnv(reward_shaping_env),
                gamma=params["gamma"],
            )
            stats_envs.append(self.passive_stats_env)

            # Make it move with the original env
            self.env = UnshapedEnvWrapper(
                shaped_env=cast(RewardShapingWrapper, self.env),
                unshaped_env=cast(UnshapedEnv, self.passive_stats_env),
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
        data = []
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
