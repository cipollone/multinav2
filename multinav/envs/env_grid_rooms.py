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
import logging
from typing import Any, Mapping, Optional

from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import actions, configurations
from gym_sapientino.core.types import Colors
from gym_sapientino.core.types import id2color as room2color
from temprl.types import Interpretation

from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.utils import SingleAgentWrapper

logger = logging.getLogger(__name__)

# Global def
color2int = {str(c): i for i, c in enumerate(list(Colors))}
int2color = {i: c for c, i in color2int.items()}


class GridRoomsFluents(FluentExtractor):
    """Define propositions for GridRooms."""

    def __init__(self, map_config: str):
        """Initialize.

        :param map_config: the charmap of a sapientino map. This is used to collect
            the number of rooms.
        """
        charset = set(map_config)
        charset -= {'|', '#', ' ', '\n', '\r'}
        self.rooms = charset
        self.n_rooms = len(charset)
        self.fluents = {str(room2color[room]) for room in self.rooms}

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def __call__(self, obs: Mapping[str, Any], _action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from a SapientinoDictSpace
            wrapped in SingleAgentWrapper.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        color = int2color[obs["color"]]
        assert color not in ("blank", "wall"), (
            "Colors should be everywhere because they fill rooms")
        return {color}


def make(params: Mapping[str, Any], log_dir: Optional[str] = None):
    """Make the grid_rooms environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: a gym Environemnt.
    """
    # Define the robot
    agent_configuration = configurations.SapientinoAgentConfiguration(
        initial_position=params["initial_position"],
        commands=actions.GridCommand,
    )

    # Define the environment
    configuration = configurations.SapientinoConfiguration(
        (agent_configuration,),
        grid_map=params["map"],
        reward_outside_grid=params["reward_outside_grid"],
        reward_duplicate_beep=params["reward_duplicate_beep"],
        reward_per_step=params["reward_per_step"],
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    fluent_extractor = GridRoomsFluents(map_config=params["map"])

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
        must_load=True,
    )

    # Time limit (this should be before reward shaping)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    # Reward shaping on previous envs
    if params["shaping"]:
        raise AssertionError()  # TODO

    # Choose the environment features
    env = GridRobotFeatures(env)

    return env
