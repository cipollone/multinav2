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

"""This environment is a sapientino with continuous movements.

Internally, we can use the same simulator than the grid sapientino,
but we extract different features. This file defines a specific environment
configuration, map, and features extraction. This is the environment used for
the experiments. Some parameters can be controlled through arguments,
others can be edited here.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Set

from flloat.semantics import PLInterpretation
from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.core.types import color2int

from multinav.restraining_bolts.automata import make_sapientino_goal_with_automata
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper

"""This is the map of the grid."""
_sapientino_map_str = """\
|       |
|       |
| rgb   |
|       |"""
_sapientino_colors_in_map = {"red", "green", "blue"}


class Fluents:
    """Define the propositions in this specific environment."""

    def __init__(self, colors_set: Set[str]):
        """Initialize.

        :param colors_set: a set of colors among the ones used by sapientino;
            this will be the set of fluents to valuate.
        """
        self._color2int = {c.value: i for c, i in color2int.items()}
        self._int2color = {i: c for c, i in self._color2int.items()}
        self.colors_set = colors_set
        if not self.colors_set.issubset(self._color2int):
            raise ValueError(str(colors_set) + " contains invalid colors")

    def valuate(self, obs: Dict[str, float], action) -> PLInterpretation:
        """Compute a propositional interpretation.

        This function respects the interface defined in
        temprl.wrapper.TemporalGoal.
        :param obs: env observation; assuming a dict observation space.
        :param action: env action.
        :return: a propositional interpretation
        """
        beeps = obs.get("beep") > 0
        if not beeps:
            fluents = {}
        else:
            color_id = obs.get("color")
            color_name = self._int2color[color_id]
            if color_name == "blank":
                fluents = {}
            else:
                if color_name not in self.colors_set:
                    raise RuntimeError("Unexpected color: " + color_name)
                fluents = {color_name}
            print(fluents)
            return PLInterpretation(fluents)


def make_env(params: Dict[str, Any]):
    """Return sapientino continuous state environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :return: an object that respects the gym.Env interface.
    """
    # Define the robot
    agent_configuration = SapientinoAgentConfiguration(
        continuous=True,
        initial_position=params["initial_position"],
    )

    # Define the map
    map_file = Path(tempfile.mktemp(suffix=".txt"))
    map_file.write_text(_sapientino_map_str)

    # Define the environment
    configuration = SapientinoConfiguration(
        [agent_configuration],
        path_to_map=map_file,
        reward_per_step=-0.01,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        acceleration=params["acceleration"],
        angular_acceleration=params["angular_acceleration"],
        max_velocity=params["max_velocity"],
        min_velocity=params["min_velocity"],
        max_angular_vel=params["angular_acceleration"],
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    fluents = Fluents(colors_set=_sapientino_colors_in_map)

    # Define the temporal goal
    tg = make_sapientino_goal_with_automata(
        colors=fluents.colors_set,
        fluent_extractor=fluents.valuate,
        reward=1.0,
    )
    env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    return env
