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
but we extract different features.
"""

from pathlib import Path

from flloat.semantics import PLInterpretation
from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)

from multinav.restraining_bolts.automata import make_sapientino_goal_with_automata
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper

# TODO: add the distinction between with and without RB
# TODO: move fluent extraction ouside of this fn.


def make_sapientino_cont_env(params):
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
    # Define the environment
    configuration = SapientinoConfiguration(
        [agent_configuration],
        path_to_map=Path("inputs/sapientino-map.txt"),
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
    colors = ["red", "green", "blue"]

    def extract_sapientino_fluents(obs, action):
        """Extract Sapientino fluents."""
        is_beep = obs.get("beep") > 0
        color_id = obs.get("color")
        if is_beep and 0 <= color_id - 1 < len(colors):
            color = colors[color_id - 1]
            fluents = {color} if color in colors else set()
        else:
            fluents = set()
        return PLInterpretation(fluents)

    # Define the temporal goal
    tg = make_sapientino_goal_with_automata(colors, extract_sapientino_fluents)
    env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])
    print("Temporal goal:", tg._formula)

    return env
