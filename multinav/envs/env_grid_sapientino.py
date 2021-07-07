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
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set, cast

from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.core.types import Colors, color2id
from temprl.types import FluentExtractor, Interpretation

from multinav.algorithms.agents import ValueFunctionModel
from multinav.envs import sapientino_defs, temporal_goals
from multinav.envs.env_abstract_sapientino import OfficeFluents as AbstractOfficeFluents
from multinav.helpers.callbacks import Callback
from multinav.helpers.reward_shaping import StateH, StateL, ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.utils import CallbackWrapper, SingleAgentWrapper


class GridSapientinoRewardShaper(ValueFunctionRS):
    """Reward shaper for grid Sapientino."""

    def __value_function_callable(self, state):
        return self.value_function_table[state]

    def __mapping_function(self, state):
        agent_state, automata_states = state[0], state[1:]
        color = agent_state["color"]
        return (color,) + tuple(*automata_states)

    def __init__(self, value_function_table, gamma):
        """Initialize the Sapientino reward shaper."""
        self.value_function_table = value_function_table
        ValueFunctionRS.__init__(
            self,
            value_function=self.__value_function_callable,
            mapping_function=self.__mapping_function,
            gamma=gamma,
            zero_terminal_state=False,
        )


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


class Fluents:
    """Define the propositions in the sapientino environment.

    This fluents evaluator works for any environment built on
    gym_sapientino repository.
    """

    def __init__(self, colors_set: Set[str]):
        """Initialize.

        :param colors_set: a set of colors among the ones used by sapientino;
            this will be the set of fluents to evaluate.
        """
        self.fluents = colors_set
        if not self.fluents.issubset(sapientino_defs.color2int):
            raise ValueError(str(colors_set) + " contains invalid colors")

    def __call__(self, obs: Dict[str, Any], _: int) -> Interpretation:
        """Respects temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractSapientino` environment.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        beeps = obs["beep"] > 0
        if not beeps:
            true_fluents = set()  # type: Set[str]
        else:
            color_id = obs["color"]
            color_name = sapientino_defs.int2color[color_id]
            if color_name == "blank":
                true_fluents = set()
            else:
                if color_name not in self.fluents:
                    raise RuntimeError("Unexpected color: " + color_name)
                true_fluents = {color_name}

        return {f for f in self.fluents if f in true_fluents}


class OfficeFluents:
    """Same as OfficeFluents in abstract office sapientino, but for grid.

    It requires the ResetCallback to be called in order to work properly.
    """

    def __init__(self, n_rooms: int, seed: int):
        """Initialize."""
        self._inner = AbstractOfficeFluents(n_rooms=n_rooms, seed=seed)

    def _new_episode(self):
        """Update for a new episode."""
        # Sample once per episode
        self._samples = self._inner._rng.integers(
            0, 2, size=(self._inner._n_rooms, 2))

    def __call__(self, obs: Dict[str, Any], _: int) -> Interpretation:
        """Respects temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractSapientino` environment.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        color_id = obs["color"]
        color_name = sapientino_defs.int2color[color_id]
        beep = obs["beep"] > 0

        fluents = set()
        if color_name != "blank":
            fluents.add(self._inner._colors_to_room[color_name])
        if beep:
            fluents.add("bip")

        # Doors and people are not in the observation. Valuate at random
        if color_id != 0:
            room_id = int(color_id / 2) - 1
            assert 0 <= room_id < self._inner._n_rooms
            if self._samples[room_id, 0] == 1:
                fluents.add("closed")
            if self._samples[room_id, 1] == 1:
                fluents.add("person")

        return {f for f in self._inner.fluents if f in fluents}

    class ResetCallback(Callback):
        """Use this callback together with this fluents evaluator."""

        def __init__(self, fluents: "OfficeFluents"):
            """Initialize."""
            self._fluents = fluents

        def _on_reset(self, obs) -> None:
            self._fluents._new_episode()

        def _on_step(self, action, obs, reward, done, info) -> None:
            pass


def abs_sapientino_shaper(path: str, gamma: float) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `AbstractSapientinoTemporalGoal` then
    it uses it to compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint for `AbstractSapientinoTemporalGoal`
    :param gamma: RL discount factor.
    :return: reward shaper to apply.
    """
    # AbstractSapientinoTemporalGoal's agent is a ValueFunctionModel
    agent = ValueFunctionModel.load(path=path)

    # Define mapping
    def _map(state: StateL) -> StateH:
        agent_state, automata_states = state[0], state[1:]
        color = agent_state["color"]
        # Wall (color == 1) is not mapped to anything here
        if color != 0:
            color -= 1
        return (color,) + tuple(*automata_states)

    # Shaper
    shaper = ValueFunctionRS(
        value_function=lambda s: agent.value_function[s],
        mapping_function=_map,
        gamma=1.0,   # NOTE: this is intentional
        zero_terminal_state=False,  # NOTE ^
    )

    return shaper


def make(params: Dict[str, Any], log_dir: Optional[str] = None):
    """Make the sapientino grid state environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: a gym Environemnt.
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
        (agent_configuration,),
        path_to_map=map_file,
        reward_per_step=params["reward_per_step"],
        reward_outside_grid=params["reward_outside_grid"],
        reward_duplicate_beep=params["reward_duplicate_beep"],
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    n_rooms = sapientino_defs.sapientino_n_rooms
    fluent_extractor = OfficeFluents(n_rooms=n_rooms, seed=params["seed"])

    # Update fluents
    env = CallbackWrapper(env, OfficeFluents.ResetCallback(fluent_extractor))

    # Apply temporal goals to this env
    env = temporal_goals.with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=cast(FluentExtractor, fluent_extractor),
        log_dir=log_dir,
        must_load=True,
    )

    # Time limit (this should be before reward shaping)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    # Reward shaping on previous envs
    if params["shaping"]:
        abs_shaper = abs_sapientino_shaper(
            path=params["shaping"],
            gamma=params["gamma"],
        )
        env = RewardShapingWrapper(env, reward_shaper=abs_shaper)

    # Choose the environment features
    env = GridRobotFeatures(env)

    return env
