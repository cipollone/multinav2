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

import time
from typing import Any, Dict, Optional, Tuple

import gym

from multinav.algorithms.agents import AgentModel
from multinav.helpers.gym import Action, Done, Reward, State
from multinav.training import TrainerSetup

GymStep = Tuple[State, Reward, Done, Dict[str, Any]]


def test(
    params: Dict[str, Any],
    load: str,
    interactive: bool = False,
    render: bool = False,
):
    """Test an agent.

    :param params: for examples on these parameters see the project repository
        in files under inputs/
    :param load: path to agent checkpoint file.
    :param interactive: watch step by step and decide actions.
    :param render: render the image and and delays.
    """
    # Make
    train_setup = TrainerSetup(params)

    # Load
    agent = train_setup.agent.load(load)

    # Start
    Tester(env=train_setup.env, model=agent, interactive=interactive, render=render).test()


class Tester:
    """Define the testing loop."""

    def __init__(
        self,
        env: gym.Env,
        model: AgentModel,
        interactive: bool = False,
        render: bool = False,
    ):
        """Initialize."""
        self.env = env
        self.model = model
        self._interactive = interactive
        self._render = render

    def test(self):
        """Test loop."""
        # Episodes
        for _ in range(10):

            # Init episode
            obs = self.env.reset()
            reward = 0.0
            done = False
            info = dict()

            while not done:
                # Render
                if self._render:
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
