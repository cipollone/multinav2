# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#
"""Reward shaping wrapper."""
from pathlib import Path

import gym
from gym.spaces import Discrete, MultiDiscrete
from gym_sapientino.core.types import Colors, color2id

from multinav.helpers.gym import RewardShaper


class GridSapientinoRewardShaper(RewardShaper):
    """Reward shaper for grid Sapientino."""

    def _value_function_callable(self, state):
        return self.value_function_table[state]

    def _mapping_function(self, state):
        agent_state, automata_states = state[0], state[1:]
        color = agent_state["color"]
        return (color,) + tuple(*automata_states)

    def __init__(self, value_function_table):
        """Initialize the Sapientino reward shaper."""
        self.value_function_table = value_function_table
        super().__init__(self._value_function_callable, self._mapping_function)


class RobotFeatures(gym.Wrapper):
    """
    Wrapper for Sapientino with temporal goal.

    This wrappers extracts coordinate x and y from
    the dictionary space of SapientinoDictSpace,
    and flattens the automata spaces.
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper."""
        super().__init__(env)

        spaces = self.observation_space.spaces  # type: ignore
        assert len(spaces) == 2
        robot_space, automata_space = spaces
        assert isinstance(automata_space, MultiDiscrete)
        assert isinstance(robot_space, gym.spaces.dict.Dict)

        x_space: Discrete = robot_space.spaces["x"]
        y_space: Discrete = robot_space.spaces["y"]
        self.observation_space = MultiDiscrete(
            [x_space.n, y_space.n, *automata_space.nvec]
        )

    def _process_state(self, state):
        """Process the observation."""
        robot_state, automata_states = state[0], state[1]
        new_state = (robot_state["x"], robot_state["y"], *automata_states)
        return new_state

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        new_state = self._process_state(state)
        return new_state, reward, done, info

    def reset(self, **_kwargs):
        """Reset."""
        state = super().reset(**_kwargs)
        return self._process_state(state)


def generate_grid(
    nb_colors: int, output_file: Path, nb_rows: int = 5, space_between_colors: int = 2
):
    """
    Generate a grid.

    :param nb_colors: the number of colors.
    :param output_file: path where to write
    :param nb_rows: number of rows.
    :param space_between_colors: spaces between colors.
    :return: None
    """
    # nb colors + spaces between colors + first and last column.
    nb_columns = nb_colors + space_between_colors * (nb_colors - 1) + 2
    cells = []

    row = " " * nb_columns
    cells += [row] * (nb_rows // 2)
    cells += [
        " "
        + (" " * space_between_colors).join(
            map(lambda x: color2id[x], list(Colors)[1 : nb_colors + 1])  # noqa: ignore
        )
        + " "
    ]
    cells += [row] * (nb_rows // 2)

    content = "\n".join(cells)
    output_file.write_text(content)
