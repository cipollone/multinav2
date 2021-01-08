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
from typing import Any, Dict, Optional

import gym
from stable_baselines.common import BaseRLModel

from multinav.algorithms.agents import AgentModel
from multinav.envs import (
    env_abstract_sapientino,
    env_cont_sapientino,
    env_grid_sapientino,
    env_ros_controls,
)
from multinav.helpers.misc import prepare_directories
from multinav.training import TrainQ, TrainStableBaselines, TrainValueIteration
from multinav.wrappers.utils import AbstractSapientinoRenderer


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
        # Make env
        env = env_ros_controls.make(params=params)
        # Make and load agent
        model = TrainStableBaselines(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        ).model

    elif env_name == "sapientino-cont":
        # Make env
        env = env_cont_sapientino.make(params=params)
        # Make and load agent
        model = TrainStableBaselines(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        ).model

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
    tester = Tester(env=env, model=model)

    # Start
    tester.test()


class Tester:
    """Define the testing loop."""

    def __init__(self, env: gym.Env, model: AgentModel):
        """Initialize."""
        self.env = env
        self.model = model
        self._is_stable_baselines_model = isinstance(model, BaseRLModel)

    def test(self):
        """Test loop."""
        # Episodes
        for _ in range(100):

            # Init episode
            obs = self.env.reset()
            done = False

            while not done:
                # Render
                self.env.render()

                # Compute action
                action = self.model.predict(obs)
                if self._is_stable_baselines_model:
                    assert action[1] is None
                    action = action[0]

                # Move env
                obs, _, done, _ = self.env.step(action)

                # Let us see the screen
                time.sleep(0.1)
