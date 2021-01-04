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

Internally, we can use the same simulator as the grid sapientino,
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

from multinav.envs import sapientino_defs
from multinav.envs.base import AbstractFluents
from multinav.envs.temporal_goals import SapientinoGoal
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper


# TODO: move to sapientino_defs and rename sapientino_defs
class Fluents(AbstractFluents):
    """Define the propositions in this specific environment.

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

    def evaluate(self, obs: Dict[str, float], action: int) -> PLInterpretation:
        """Respects AbstractFluents.evaluate."""
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
        return PLInterpretation(true_fluents)


def make(params: Dict[str, Any]):
    """Make the sapientino continuous state environment.

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
    map_file.write_text(sapientino_defs.sapientino_map_str)

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
    fluents = Fluents(colors_set=set(sapientino_defs.sapientino_color_sequence))

    # Define the temporal goal
    tg = SapientinoGoal(
        colors=sapientino_defs.sapientino_color_sequence,
        fluents=fluents,
        reward=params["tg_reward"],
    )
    env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    return env
