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
"""Wrappers specific to Sapientino."""
from abc import abstractmethod

import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple

from multinav.helpers.gym import combine_boxes


class AbstractRobotFeatures(gym.Wrapper):
    """
    Abstract wrapper for features extraction in Sapientino with temporal goal.

    This wrappers extracts specific fields from
    the dictionary space of SapientinoDictSpace,
    and flattens the automata spaces due to the temporal wrapper.
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.

        :param env: the environment to wrap.
        """
        super().__init__(env)

        spaces = env.observation_space.spaces  # type: ignore
        assert len(spaces) == 2
        self.robot_space, self.automata_space = spaces
        assert isinstance(self.automata_space, MultiDiscrete)
        assert isinstance(self.robot_space, gym.spaces.dict.Dict)

        self.observation_space = self.compute_observation_space()

    @abstractmethod
    def compute_observation_space(self) -> gym.Space:
        """Get the observation space."""

    @abstractmethod
    def _process_state(self, state):
        """Process the observation."""

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        new_state = self._process_state(state)
        return new_state, reward, done, info

    def reset(self, **_kwargs):
        """Reset."""
        state = super().reset(**_kwargs)
        return self._process_state(state)


class GridRobotFeatures(AbstractRobotFeatures):
    """
    Wrapper for features extraction in grid Sapientino with temporal goal.

    This wrappers extracts specific fields from
    the dictionary space of SapientinoDictSpace,
    and flattens the automata spaces due to the temporal wrapper.
    """

    def compute_observation_space(self) -> gym.Space:
        """Get the observation space."""
        x_space: Discrete = self.robot_space.spaces["discrete_x"]
        y_space: Discrete = self.robot_space.spaces["discrete_y"]
        return MultiDiscrete([x_space.n, y_space.n, *self.automata_space.nvec])

    def _process_state(self, state):
        """Process the observation."""
        robot_state, automata_states = state[0], state[1]
        new_state = (
            robot_state["discrete_x"],
            robot_state["discrete_y"],
            *automata_states,
        )
        return new_state


class ContinuousRobotFeatures(AbstractRobotFeatures):
    """Wrapper for features extraction in continuous Sapientino with temporal goal."""

    def compute_observation_space(self) -> gym.Space:
        """Get the observation space."""
        x_space: Box = self.robot_space.spaces["x"]
        y_space: Box = self.robot_space.spaces["y"]
        ang_velocity_space: Box = self.robot_space.spaces["ang_velocity"]

        # Try with cos, sin, instead
        cos_space = Box(-1, 1, shape=[1])
        sin_space = Box(-1, 1, shape=[1])

        # Try with derivative in components
        dx_space = Box(-float("inf"), float("inf"), shape=[1])
        dy_space = Box(-float("inf"), float("inf"), shape=[1])

        # Join sapientino features
        sapientino_space = combine_boxes(
            x_space,
            y_space,
            cos_space,
            sin_space,
            dx_space,
            dy_space,
            ang_velocity_space,
        )

        return GymTuple((sapientino_space, self.automata_space))

    def _process_state(self, state):
        """Process the observation."""
        robot_state, automata_states = state

        cos = np.cos(robot_state["angle"])
        sin = np.sin(robot_state["angle"])

        sapientino_state = np.array(
            [
                robot_state["x"],
                robot_state["y"],
                cos,
                sin,
                robot_state["velocity"] * cos,
                robot_state["velocity"] * sin,
                robot_state["ang_velocity"],
            ]
        )
        return sapientino_state, automata_states
