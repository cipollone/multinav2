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
"""Test on environment."""

import json
import time
from typing import Any, Dict, Optional, Tuple

import gym
from stable_baselines.common import BaseRLModel

from multinav.algorithms.agents import AgentModel
from multinav.envs import (
    env_abstract_sapientino,
    env_cont_sapientino,
    env_grid_sapientino,
    env_ros_controls,
)
from multinav.helpers.gym import Action, Done, Reward, State
from multinav.helpers.misc import prepare_directories
from multinav.training import TrainQ, TrainStableBaselines, TrainValueIteration
from multinav.wrappers.utils import AbstractSapientinoRenderer

GymStep = Tuple[State, Reward, Done, Dict[str, Any]]


def test(
    env_name: str,
    json_params: Optional[str] = None,
    cmd_params: Optional[Dict[str, Any]] = None,
):
    """Train an agent on the ROS environment.

    Parameter "resume_file" should point to an existing checkpoint; the other
    parameters should be the ones used for training.
    :param env_name: the environment id (see ``multinav --help``)
    :param json_params: the path (str) of json file of parameters.
    :param cmd_params: optional command line parameters. These are meant to
        override json_params.
    """
    # Settings
    if not json_params:
        raise TypeError("You must supply the parameters of the preivous training.")
    with open(json_params) as f:
        params = json.load(f)  # type: Dict[str, Any]
    if cmd_params:
        params.update(cmd_params)

    # Init dirs
    resuming = bool(params["resume_file"])
    if not resuming:
        raise RuntimeError("Must resume from a checkpoint in order to test")
    model_path, log_path = prepare_directories(
        env_name=env_name,
        no_create=True,
    )

    if env_name == "ros":
        # Load agent and make env
        trainer = TrainStableBaselines(
            env=env_ros_controls.make(params=params),
            params=params,
            model_path=model_path,
            log_path=log_path,
        )
        model = trainer.model
        env = model.env
        # Freeze normalization weights
        trainer.normalized_env.set_training(False)

    elif env_name == "sapientino-cont":
        # Load agent and make env
        trainer = TrainStableBaselines(
            env=env_cont_sapientino.make(params=params),
            params=params,
            model_path=model_path,
            log_path=log_path,
        )
        model = trainer.model
        env = model.env
        # Freeze normalization weights
        trainer.normalized_env.set_training(False)

    elif env_name == "sapientino-grid":
        resume_file = params.pop("resume_file")
        # Make env
        env = env_grid_sapientino.make(params=params)
        # Make agent
        model = TrainQ(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        ).agent
        # Load agent
        model = model.load(resume_file)

    elif env_name == "sapientino-abs":
        resume_file = params.pop("resume_file")
        # Make env
        env = env_abstract_sapientino.make(params=params, log_dir=log_path)
        env = AbstractSapientinoRenderer(env)
        # Make agent
        model = TrainValueIteration(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        ).agent
        # Load agent
        model = model.load(resume_file)
    else:
        raise RuntimeError("Environment not supported")

    # Same testing loop for all
    tester = Tester(
        env=env,
        model=model,
        interactive=params["interactive"],
        deterministic=params["deterministic"],
    )

    # Start
    tester.test()


class Tester:
    """Define the testing loop."""

    def __init__(
        self,
        env: gym.Env,
        model: AgentModel,
        interactive: bool = False,
        deterministic: bool = False,
    ):
        """Initialize."""
        self.env = env
        self.model = model
        self._interactive = interactive
        self._deterministic = deterministic
        self._is_stable_baselines_model = isinstance(model, BaseRLModel)

        if self._is_stable_baselines_model:
            self._wrap_baserlmodel_predict()

    def _wrap_baserlmodel_predict(self):
        """Wrap instance BaseRLModel.predict appropriately."""
        original_predict = self.model.predict

        def _predict_fn(observation):
            ret = original_predict(observation, deterministic=self._deterministic)
            assert ret[1] is None
            return ret[0]

        self.model.predict = _predict_fn

    def test(self):
        """Test loop."""
        # Episodes
        for _ in range(10):

            # Init episode
            obs = self.env.reset()
            reward = None
            done = False
            info = None

            while not done:
                # Render
                self.env.render()

                # Compute action
                action = self.model.predict(obs)

                # Maybe interact
                if self._interactive:
                    action = self._interact((obs, reward, done, info), action)
                    if action < 0:
                        break

                # Move env
                obs, reward, done, info = self.env.step(action)

                # Print
                if self._interactive and done:
                    self._interact((obs, reward, done, info), action, False)

                # Let us see the screen
                time.sleep(0.1)

    def _interact(
        self, data: GymStep, action: Action, ask: Optional[bool] = True
    ) -> Action:
        """Interact with user.

        The function shows some data, then asks for an action on the command
        line.
        :param stata: the last tuple returned by gym environment.
        :param action: the last action selected by the agent.
        :param ask: whether we should ask the use or just print to cmd line.
        :return: the action to perform; defaults to input action.
        """
        print("Env step")
        print("  Observation:", data[0])
        print("       Reward:", data[1])
        print("         Done:", data[2])
        print("        Infos:", data[3])
        if not ask:
            return action

        act = input(
            "Action in [-1, {}] (default {})? ".format(
                self.env.action_space.n - 1, action
            )
        )
        if act is not None and act != "":
            action = int(act)
        if action < 0:
            print("Reset")

        return action
