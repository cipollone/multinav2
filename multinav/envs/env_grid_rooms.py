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
"""Grid control on rooms environment."""
import logging
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import actions, configurations
from gym_sapientino.core.types import Colors, color2id, id2color
from temprl.types import Interpretation

from multinav.algorithms.agents import QFunctionModel
from multinav.envs.env_abstract_rooms import AbstractRooms
from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.reward_shaping import StateH, ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper, RewardShift
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.utils import FailProbability, SingleAgentWrapper

logger = logging.getLogger(__name__)

# Global def
color2int = {str(c): i for i, c in enumerate(list(Colors))}
int2color = {i: c for c, i in color2int.items()}


class RoomsFluents(FluentExtractor):
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
        self.fluents = {str(id2color[room]) for room in self.rooms}

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


class Grid2Abs:
    """Transform features of grid_rooms to AbstractRooms."""

    def __init__(self, rooms_connectivity: Sequence[Sequence[str]]):
        """Initialize.

        :param rooms_connectivity: see AbstractRooms.
        """
        self._abs_env = AbstractRooms(
            rooms_connectivity=rooms_connectivity,
            initial_room=rooms_connectivity[0][0],
            p_failure=0.0,
        )

    def __call__(self, obs: Mapping[str, Any]):
        """Convert to abstract."""
        color = int2color[obs["color"]]
        assert color not in ("blank", "wall"), (
            "Colors should be everywhere because they fill rooms")
        abs_state = self._abs_env._room2id[color2id[Colors(color)]]
        return abs_state


def abs_rooms_shaper(
    path: str,
    gamma: float,
    return_invariant: bool,
    rooms_connectivity: List[List[str]],
) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `AbstractRooms` then it uses it to
    compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint.
    :param gamma: discount factor to apply for shaping.
    :param return_invariant: if true, we apply classic return-invariant reward shaping.
        We usually want this to be false.
    :param rooms_connectivity: see AbstractRooms.
    :return: reward shaper to apply.
    """
    # Trained agent on abstract environment
    agent = QFunctionModel.load(path=path)

    # Map
    grid_to_abs = Grid2Abs(rooms_connectivity=rooms_connectivity)

    def map_with_temporal_goals(state: Tuple[Mapping[str, Any], list]) -> StateH:
        obs = grid_to_abs(state[0])
        qs = state[1]
        state1 = (obs, *qs)
        logger.debug("Mapped state: %s", state1)
        return state1

    # Shaper
    shaper = ValueFunctionRS(
        value_function=lambda s: agent.q_function[s].max(),
        mapping_function=map_with_temporal_goals,
        gamma=gamma,
        zero_terminal_state=return_invariant,
    )

    return shaper


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

    # Fail probability
    if params["fail_p"] > 0:
        env = FailProbability(env, fail_p=params["fail_p"], seed=params["seed"])

    # Define the fluent extractor
    fluent_extractor = RoomsFluents(map_config=params["map"])

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
        must_load=True,
    )

    # Reward shift
    if params["reward_shift"] != 0:
        env = RewardShift(env, params["reward_shift"])

    # Time limit (this should be before reward shaping)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    # Reward shaping on previous envs
    if params["shaping"]:
        abs_shaper = abs_rooms_shaper(
            path=params["shaping"],
            gamma=params["shaping_gamma"],
            return_invariant=params["return_invariant"],
            rooms_connectivity=params["rooms_connectivity"],
        )
        env = RewardShapingWrapper(env, reward_shaper=abs_shaper)

    # Choose the environment features
    env = GridRobotFeatures(env)

    return env
