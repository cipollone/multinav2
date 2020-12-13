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
