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
import random
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, cast

import gym
import numpy as np
import ray
import tensorflow as tf
import yaml
from gym import Env
from matplotlib import pyplot as plt
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks

from multinav.algorithms.agents import AgentModel, QFunctionModel, RllibAgentModel
from multinav.algorithms.delayed_q import DelayedQAgent
from multinav.algorithms.q_learning import QLearning
from multinav.envs.envs import EnvMaker
from multinav.helpers.callbacks import CallbackList as CustomCallbackList
from multinav.helpers.callbacks import FnCallback, SaverCallback
from multinav.helpers.general import QuitWithResources
from multinav.helpers.gym import find_wrapper
from multinav.helpers.rllib import DiscountedRewardLogger
from multinav.wrappers.reward_shaping import (
    RewardShapingWrapper,
    UnshapedEnv,
    UnshapedEnvWrapper,
)
from multinav.wrappers.utils import CallbackWrapper, MyStatsRecorder


class TrainerSetup:
    """Train an agent."""

    def __init__(self, params: Dict[str, Any], agent_only=False):
        """Initialize.

        :param params: for examples on these parameters see the project repository
            in files under inputs/
        :param agent_only: if true, only load definitions but do not start.
        """
        # Get options
        self.params = params
        self.alg_params: Dict[str, Any] = params["algorithm"]["params"]
        self.env_params: Dict[str, Any] = params["environment"]["params"]
        self.env_name = self.env_params["env"]

        common_params = {
            "seed": params["seed"],
            "logs-dir": params["logs-dir"],
            "model-dir": params["model-dir"],
            "gamma": (
                self.alg_params["gamma"] if "gamma" in self.alg_params
                else self.alg_params["config"]["gamma"]
            ),
            "run-id": params["run-id"],
        }
        self.env_params.update(common_params)
        self.alg_params.update(common_params)

        # Trainer for tabular environments
        self.trainer: Trainer
        self.agent: AgentModel
        if "0" not in self.env_name:  # Because 0 is for continuous environments
            algorithm = self.alg_params["algorithm"]
            if algorithm == "Q":
                TrainerClass = TrainQ
            elif algorithm == "DelayedQ":
                TrainerClass = TrainDelayedQ
            else:
                raise ValueError("Error in training algorithm name")

            self.trainer = TrainerClass(
                env=EnvMaker(self.env_params).env,
                params=self.alg_params,
                agent_only=agent_only,
            )
            self.agent = self.trainer.agent
            self.passive_agent = self.trainer.passive_agent
            self.env_maker = lambda: EnvMaker(self.env_params)

        # Trainer for continuous environments
        else:
            self.trainer = TrainRllib(
                env_class=EnvMaker,
                alg_params=self.alg_params,
                env_params=self.env_params,
                agent_only=agent_only,
            )
            self.agent = self.trainer.agent
            self.env_maker = lambda: self.agent.trainer.workers.local_worker().env
            # TODO: active passive?

        # Closing env resources
        if "stop_script" in params:
            def _env_stopper():
                print("Stopping")
                subprocess.run(params["stop_script"])
                print("Stopped")
            QuitWithResources.add("env_stopper", _env_stopper)

    def train(self):
        """Start training."""
        self.trainer.train()


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
        agent_only: bool = False,
    ):
        """Initialize.

        :param env: discrete-state gym environment.
        :param params: dict of parameters. See `default_parameters`.
        :param agent_only: if True, it defines or loads the agent(s),
            then stops. The environment is not used. For training,
            this must be false.
        """
        model_path = params["model-dir"]
        log_path = params["logs-dir"]

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

        # Q function used as exploration policy
        self.exploration_Q = None
        if params["exploration_policy"]:
            self.exploration_Q = QFunctionModel.load(path=params["exploration_policy"]).q_function

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
            alpha=params["learning_rate"],
            eps=params["q_eps"],
            gamma=params["gamma"],
            learning_rate_decay=True,
            learning_rate_end=params["learning_rate_end"],
            epsilon_decay=True,
            epsilon_end=params["epsilon_end"],
            epsilon_end_decay=params.get("epsilon_end_decay", params["total_timesteps"]),
            action_bias=self.biased_Q,
            action_bias_eps=params["action_bias_eps"],
            exploration_policy=self.exploration_Q,
            initial_Q=self.agent.q_function if self._reinitialized else None,
            active_passive_agents=params["active_passive_agents"],
            passive_reward_getter=original_reward_getter,
            initial_passive_Q=(
                self.passive_agent.q_function if self._reinitialized else None
            ),
            seed=params["seed"],
            rollout_interval=params["rollout_interval"],
            rollout_episodes=params["rollout_episodes"],
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
        draw_lines = [None for _ in range(len(self._log_properties))]

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
        try:
            self.learner.learn(max_steps=self.params["total_timesteps"])

        # Save
        finally:
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

        # Save evaluation metrics to file
        if hasattr(self.learner, "eval_stats"):
            eval_file = os.path.join(self._log_path, "evaluation_log.yaml")
            with open(eval_file, "w") as f:
                yaml.dump(self.learner.eval_stats, f)


class TrainDelayedQ(TrainQ):
    """Agent and training loop for DelayedQ learning."""

    def __init__(
        self,
        env: Env,
        params: Dict[str, Any],
        agent_only: bool = False,
    ):
        """Initialize.

        :param env: discrete-state gym environment.
        :param params: dict of parameters. See `default_parameters`.
        :param agent_only: if True, it defines or loads the agent(s),
            then stops. The environment is not used. For training,
            this must be false.
        """
        model_path = params["model-dir"]
        log_path = params["logs-dir"]

        # Check
        if "resume_file" in params and params["resume_file"]:
            raise TypeError("Resuming a training is not supported here")
        if params["active_passive_agents"]:
            raise NotImplementedError("Passive agent not implemented yet")

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
        # stats_envs = [self.stats_env] TODO

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
        self.learner = DelayedQAgent(
            env=self.env,
            stats_env=self.stats_env,
            gamma=params["gamma"],
            eps1=params["eps1"],
            delta=params["delta"],
            maxr=params["maxr"],
            minr=params["minr"],
            m=params["m"],
            rollout_interval=params["rollout_interval"],
            rollout_episodes=params["rollout_episodes"],
        )

        # Link trained and saved agents
        self.agent.q_function = self.learner.Q
        # self.passive_agent.q_function = self.learner.passive_Q TODO
        self.saver.extra_model = self.passive_agent.q_function


class TrainRllib(Trainer):
    """Agent and training loop for Deep learning."""

    def __init__(
        self,
        env_class: Type[gym.Env],
        alg_params: Dict[str, Any],
        env_params: Dict[str, Any],
        agent_only: bool = False,
    ):
        """Initialize.

        :param env_class: gym Env class to be instantiated.
        :param alg_params: algorithm parameters. See Ray rllib documentation
            and example config files in this project.
        :param env_params: environment parameters.
        :param agent_only: if True, it defines or loads the agent(s),
            then stops. The environment is not used. For training,
            this must be false.
        """
        # Store
        self.env_class = env_class
        self.alg_params = alg_params
        self.env_params = env_params
        self.log_path = alg_params["logs-dir"]
        self.model_path = alg_params["model-dir"]

        # Trainer config
        self.agent_type: str = self.alg_params["agent"]
        self.agent_conf: dict = self.alg_params["config"]

        # Env configs
        self.agent_conf["env"] = self.env_class
        self.agent_conf["env_config"] = self.env_params

        # Add choices to the configuration otherwise tune won't run (see with_grid_search docstring)
        self.with_grid_search(
            conf=self.agent_conf,
            tune_conf=self.alg_params["tune"],
            just_first=agent_only,
        )
        print("tune conf: ", self.agent_conf)

        # Agent interface (used when restoring from checkpoint)
        self.agent = RllibAgentModel(self.agent_type, self.agent_conf)

        # Set seed
        seed = self.alg_params["seed"]
        random.seed(seed)
        np.random.seed(seed)  # type: ignore
        tf.random.set_seed(seed)
        self.agent_conf["seed"] = seed
        self.agent_conf["env_config"]["seed"] = seed

        if agent_only:
            return

        # Init library
        if self.alg_params["run-id"] == 0:
            ray.init(include_dashboard=True)
        else:
            ray.init(address="auto")

    def train(self):
        """Start training."""
        # Insert callbacks
        self.agent_conf["callbacks"] = MultiCallbacks([
            DefaultCallbacks,
            DiscountedRewardLogger,
        ])

        # Start via Tune
        tune.run(
            self.agent_type,
            config=self.agent_conf,
            local_dir=self.log_path,
            checkpoint_at_end=True,
            name="run",
            **self.alg_params["run"],
        )

    @staticmethod
    def with_grid_search(conf, tune_conf, just_first: bool):
        """Compose the parameters to tune in the main configuration.

        conf is any configuration of an agent (a dictionary). tune_conf is another
        configuration in which values are lists instead of single elements. Any
        list define a sequence of experiments, one for each value in the list.

        NOTE: At least one list in tune_conf is always needed (even if of size 1)
        otherwise tune library won't execute any experiment. I don't know why.

        :param just_first: insert the first element of the list into the configuration.
            This does not create a tune config, but it's useful to just initialize an agent.
        """
        # Scan dictionaries
        assert isinstance(conf, dict) and isinstance(tune_conf, dict)
        assert all((k in conf for k in tune_conf.keys())), (
            f"{tune_conf.keys()} not a subset of {conf.keys()}")

        # Scan
        for key in conf:
            if key in tune_conf:

                # If not nested dict
                if not isinstance(tune_conf[key], dict):

                    # Make search space
                    vals = tune_conf[key]
                    assert isinstance(vals, list), "'tune_conf' should contain lists"
                    old_conf_val = conf[key]
                    conf[key] = tune.grid_search(vals) if not just_first else vals[0]
                    print(f"{key}: {old_conf_val} is now {key}: {conf[key]}")

                # Else traverse
                else:
                    TrainRllib.with_grid_search(conf[key], tune_conf[key], just_first)
