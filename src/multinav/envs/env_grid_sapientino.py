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
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.core.types import Colors, color2id

from multinav.algorithms.agents import ValueFunctionModel
from multinav.envs import sapientino_defs
from multinav.envs.env_cont_sapientino import Fluents
from multinav.envs.temporal_goals import SapientinoGoal
from multinav.helpers.gym import RewardShaper, StateH, StateL
from multinav.wrappers.reward_shaping import RewardShapingWrapper
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper


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
        # TODO: updated RewardShaper


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


def _load_reward_shaper(path: str, gamma: float) -> RewardShaper:
    """Load a reward shaper.

    Loads a saved agent for `AbstractSapientinoTemporalGoal` this model is
    then used to compute the reward shaping.

    :param path: path to saved checkpoint for `AbstractSapientinoTemporalGoal`
    :param gamma: RL discount factor.
    :return: reward shaper to apply.
    """
    # AbstractSapientinoTemporalGoal is a ValueFunctionModel
    agent = ValueFunctionModel.load(path=path)

    # Define mapping
    def _map(state: StateL) -> StateH:
        # NOTE: this assumes that the automaton and ids remain the same!
        #  Maybe it should be loaded too
        agent_state, automata_states = state[0], state[1:]
        color = agent_state["color"]
        return (color,) + tuple(*automata_states)

    # Mapping
    shaper = RewardShaper(
        value_function=lambda s: agent.value_function[s],
        mapping_function=_map,
        gamma=gamma,
    )

    return shaper


def make(params: Dict[str, Any], log_dir: Optional[str] = None):
    """Make the sapientino grid state environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: an object that respects the gym.Env interface.
    """
    # Define the robot
    agent_configuration = SapientinoAgentConfiguration(
        continuous=False,
        differential=False,
        initial_position=params["initial_position"],
    )

    # Define the map
    map_file = Path(tempfile.mktemp(suffix=".txt"))
    map_file.write_text(sapientino_defs.sapientino_map_str)

    # Define the environment
    configuration = SapientinoConfiguration(
        [agent_configuration],
        path_to_map=map_file,
        reward_per_step=-0.01,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    fluents = Fluents(colors_set=set(sapientino_defs.sapientino_color_sequence))

    # Define the temporal goal
    tg = SapientinoGoal(
        colors=sapientino_defs.sapientino_color_sequence,
        fluents=fluents,
        reward=params["tg_reward"],
        save_to=os.path.join(log_dir, "reward-dfa.dot") if log_dir else None,
    )
    env = MyTemporalGoalWrapper(env, [tg])

    # Time limit (this should be before reward shaping)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    # Maybe apply reward shaping
    if params["shaping"]:
        reward_shaper = _load_reward_shaper(
            path=params["shaping"],
            gamma=params["gamma"],
        )
        env = RewardShapingWrapper(env, reward_shaper=reward_shaper)

    # Final features
    env = GridRobotFeatures(env)

    return env
