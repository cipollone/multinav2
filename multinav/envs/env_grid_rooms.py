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
from gym_sapientino.core.types import Colors, color2id, color2int, id2color
from temprl.types import Interpretation

from multinav.algorithms.agents import QFunctionModel
from multinav.envs.env_abstract_rooms import AbstractPartyFluents, AbstractRooms
from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.reward_shaping import StateH, ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper, RewardShift
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.utils import FailProbability, SingleAgentWrapper

logger = logging.getLogger(__name__)

# Global def
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
        color = str(int2color[obs["color"]])
        assert color not in ("blank", "wall"), (
            "Colors should be everywhere because they fill rooms")
        return {color}


class GridPartyFluents(FluentExtractor):
    """Define propositions for Party task."""

    def __init__(
        self,
        map_config: str,
        rooms_and_locations: Sequence[Sequence[str]],
        interact_action: int,
    ):
        """Initialize.

        :param map_config: the charmap of a sapientino map.
            Each color denotes an interesting location. The rest of the space
            should be filled with another color.
        :param rooms_and_locations: see env_abstract_rooms.AbstractRoomsFluents.
        :param interact_action: the action associated to an interaction
            with the environment.
        """
        # Abstract fluents
        self.mapping = Grid2Abs(rooms_and_locations)
        self.abstract_fluents = AbstractPartyFluents(
            env=self.mapping.abstract_env,
            rooms_and_locations=rooms_and_locations,
            interact_action=interact_action,
        )

        # Store
        self.fluents = self.abstract_fluents.fluents

        # Check
        charset = set(map_config)
        charset -= {'|', '#', ' ', '\n', '\r'}
        assert self.abstract_fluents._rooms2locations.keys() <= charset, (
            f"Not all {self.abstract_fluents._rooms2locations.keys()} "
            f"are in map locations {charset}"
        )
        self._interact = interact_action

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def __call__(self, obs: Mapping[str, Any], action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from a SapientinoDictSpace
            wrapped in SingleAgentWrapper.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        abs_obs = self.mapping(obs)
        return self.abstract_fluents(abs_obs, action)


class Grid2Abs:
    """Transform features of GridRooms to AbstractRooms."""

    def __init__(self, rooms_and_locations: Sequence[Sequence[str]]):
        """Initialize.

        :param rooms_and_locations: see env_abstract_rooms.AbstractRoomsFluents.
        """
        self.abstract_env = AbstractRooms(
            rooms_connectivity=[(pair[0], pair[0]) for pair in rooms_and_locations],
            initial_room=rooms_and_locations[0][0],
            p_failure=0.0,
        )

    def __call__(self, obs: Mapping[str, Any]) -> int:
        """Convert to abstract."""
        color = int2color[obs["color"]]
        assert color != Colors.BLANK, "Colors should fill rooms"
        assert color != Colors.WALL

        abs_state = self.abstract_env._room2id[color2id[color]]
        return abs_state


def abs_rooms_shaper(
    path: str,
    gamma: float,
    return_invariant: bool,
    rooms_and_locations: Sequence[Sequence[str]],
) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `AbstractRooms` then it uses it to
    compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint.
    :param gamma: discount factor to apply for shaping.
    :param return_invariant: if true, we apply classic return-invariant reward shaping.
        We usually want this to be false.
    :param rooms_and_locations: see env_abstract_rooms.AbstractRoomsFluents.
    :return: reward shaper to apply.
    """
    # Trained agent on abstract environment
    agent = QFunctionModel.load(path=path)

    # Map
    mapping = Grid2Abs(rooms_and_locations)

    def map_with_temporal_goals(state: Tuple[Mapping[str, Any], list]) -> StateH:
        obs = mapping(state[0])
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

    # Fluents for this environment
    fluent_extractor: FluentExtractor
    if params["fluents"] == "rooms":
        fluent_extractor = GridRoomsFluents(map_config=params["map"])
    elif params["fluents"] == "party":
        fluent_extractor = GridPartyFluents(
            map_config=params["map"],
            rooms_and_locations=params["rooms_and_locations"],
            interact_action=int(actions.GridCommand.BEEP.value),
        )
    else:
        raise ValueError(params["fluents"])

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
            rooms_and_locations=params["rooms_and_locations"],
        )
        env = RewardShapingWrapper(env, reward_shaper=abs_shaper)

    # Choose the environment features
    env = GridRobotFeatures(env)

    return env
