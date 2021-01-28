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

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
from flloat.semantics import PLInterpretation
from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)

from multinav.algorithms.agents import QFunctionModel
from multinav.envs import sapientino_defs
from multinav.envs.base import AbstractFluents
from multinav.envs.temporal_goals import SapientinoGoal
from multinav.helpers.reward_shaping import StateH, StateL, ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import SingleAgentWrapper


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


def _load_reward_shaper(path: str, gamma: float) -> ValueFunctionRS:
    """Load a reward shaper.

    This loads a saved agent for `sapientino-grid` then
    it uses it to compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint for `sapientino-grid` model.
    :param gamma: RL discount factor.
    :return: reward shaper to apply.
    """
    # sapientino-grid's agent is a QFunctionModel
    agent = QFunctionModel.load(path=path)

    # Define mapping
    def _map(state: StateL) -> StateH:
        # NOTE: this assumes that the automaton and ids remain the same!
        #  Maybe it should be loaded too
        x = state[0]["discrete_x"]
        y = state[0]["discrete_y"]
        return (x, y, *state[1])

    def _valuefn(state: StateH):
        q = agent.q_function[state]
        return np.amax(q)

    # Shaper
    shaper = ValueFunctionRS(
        value_function=_valuefn,
        mapping_function=_map,
        gamma=gamma,
        zero_terminal_state=False,
    )

    return shaper


def make(params: Dict[str, Any], log_dir: Optional[str] = None):
    """Make the sapientino continuous state environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
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
        reward_per_step=params["reward_per_step"],
        reward_outside_grid=params["reward_outside_grid"],
        reward_duplicate_beep=params["reward_duplicate_beep"],
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
    env = ContinuousRobotFeatures(env)

    return env
