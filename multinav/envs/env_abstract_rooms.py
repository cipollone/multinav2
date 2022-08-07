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

"""This package contains the implementation of an 'abstract' Sapientino with teleport."""

import logging
import random
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np
from gym import Wrapper
from gym.wrappers import TimeLimit
from gym_sapientino.core.types import id2color as room2color
from temprl.types import Interpretation

from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.gym import MyDiscreteEnv, Transitions
from multinav.wrappers.reward_shaping import RewardShift
from multinav.wrappers.temprl import FlattenAutomataStates
from multinav.wrappers.utils import WithExtraAction

logger = logging.getLogger(__name__)


class AbstractRooms(MyDiscreteEnv):
    """Abstract rooms environment.

    It allows to navigate a sequence of connected rooms. Transition dynamics
    only; no reward yet.
    """

    def __init__(
        self,
        rooms_connectivity: Sequence[Sequence[str]],
        initial_room: str,
        p_failure: float = 0.0,
    ):
        """Initialize the environment.

        :param rooms_connectivity: a list of pairs of connected rooms. Rooms
            are identified with single chars.
        :param initial_room: the initial room
        :param p_failure: probability of failing a trainsition and remaining on
            the same room.
        """
        # Complete connectivity
        self.rooms_connections: Set[Tuple[str, str]] = set()
        for pair in rooms_connectivity:
            self.rooms_connections.add((pair[0], pair[1]))
            self.rooms_connections.add((pair[1], pair[0]))

        # Rooms to id
        self.rooms = {pair[0] for pair in self.rooms_connections}
        self._room2id = {char: i for i, char in enumerate(sorted(self.rooms))}
        self._id2room = {i: char for char, i in self._room2id.items()}

        logger.debug(f"rooms: {self.rooms}")
        logger.debug(f"room2id: {self._room2id}")
        logger.debug(f"n_states: {self.nb_states}")
        logger.debug(f"n_actions: {self.nb_actions}")

        # Store
        self.p_failure = p_failure
        self.initial_room = initial_room

        # Build
        model = self._make_transitions()
        isd = np.zeros(len(self.rooms))
        self.initial_state = self._room2id[initial_room]
        isd[self.initial_state] = 1.0
        super().__init__(nS=self.nb_states, nA=self.nb_actions, P=model, isd=isd)

    @property
    def nb_states(self) -> int:
        """Get the number of states."""
        return len(self.rooms)

    @property
    def nb_actions(self) -> int:
        """Get the number of actions.

        One go_to action for each other room (not always applicable).
        """
        return len(self.rooms)

    def action_goto_room(self, room_id: int) -> int:
        """Return the Go-To action associated to a location."""
        assert self._is_state(room_id), (
            f"{room_id} is not a valid room, {self.nb_states} states")
        return room_id

    def _is_state(self, state: int):
        """Check that it is a legal state."""
        return 0 <= state < self.nb_states

    def _is_action(self, action: int):
        """Check if tat is a legal action."""
        return 0 <= action < self.nb_actions

    def _make_transitions(self) -> Transitions:
        """Make the trainsition model."""
        model: Transitions = {}

        for source in self.rooms:
            source_id = self._room2id[source]

            for destination in self.rooms:
                destination_id = self._room2id[destination]

                # Skip if not connected
                if (source, destination) not in self.rooms_connections:
                    continue

                # Collect all and add
                goto_action = self.action_goto_room(destination_id)
                ok_transition = (1.0 - self.p_failure, destination_id, 0.0, False)
                fail_transition = (self.p_failure, source_id, 0.0, False)
                model.setdefault(source_id, dict()).setdefault(
                    goto_action, list()).extend([ok_transition, fail_transition])

        # Complete with no ops
        for state in range(self.nb_states):
            for action in range(self.nb_actions):
                self_loop = (1.0, state, 0.0, False)
                transitions = model[state]
                if action not in transitions:
                    transitions[action] = [self_loop]

        return model


class AbstractRoomsFluents(FluentExtractor):
    """Define propositions for AbstractRooms."""

    def __init__(self, env: AbstractRooms):
        """Initialize.

        :param env: instance of AbstractRooms.
        """
        self._env = env
        self.fluents = {str(room2color[room]) for room in self._env.rooms}

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def __call__(self, obs: int, _action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractRooms` environment.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        return {str(room2color[self._env._id2room[obs]])}


class AbstractPartyFluents(FluentExtractor):
    """Define propositions for Party task."""

    def __init__(
        self,
        env: Union[AbstractRooms, Wrapper],
        rooms_and_locations: Sequence[Sequence[str]],
        interact_action: int,
    ):
        """Initialize.

        :param env: instance of AbstractRooms.
        :param rooms_and_locations: an association between rooms and locations.
            Each row should be something like ("g", "alice"), where "g" is the
            identifier of a location and "alice" is a location.
        :param interact_action: One of the action is assumed to be an interaction.
        """
        self._id2room = env.unwrapped._id2room

        # The association between persons and locations should be consistent
        self._rooms2locations = dict()
        common_room = None
        for pair in rooms_and_locations:
            assert len(pair) == 2, f"Expected a pair, got {pair}"
            room, loc = pair
            assert room in env.rooms, "Every room should also in room connections"
            self._rooms2locations[room] = loc
            if loc == "none":
                common_room = room
        assert common_room is not None, (
            "There should be at least one pair with location 'none'"
        )

        self._interact = interact_action
        self.fluents = {"at_" + loc for loc in self._rooms2locations.values()}

        logger.debug(f"Party Fluents, rooms2locations: {self._rooms2locations}")
        logger.debug(f"Party Fluents, fluents: {self.fluents}")

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def __call__(self, obs: int, action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractRooms` environment. Rooms are interpreted as generic
            locations here.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        fluents = set()
        if action == self._interact:
            room = self._id2room[obs]
            location = self._rooms2locations[room]
            if location != "none":
                fluents.add("at_" + location)

        return fluents


class AbstractOfficeFluents(FluentExtractor):
    """Propositions for Office task.

    This class represent features such as "ouside_room1", "inside_room1",
    "door_closed", "person_detected". We are using the same environment
    AbstractRooms for this purpose, because they are conceptually equivalent.
    However, careful with the meaning of rooms. In AbstractRooms,
    a room is an alias for a color or location. Thus,
    we use two rooms/colors for each office room, one for outside_room1,
    one for inside_room1. I'll rename AbstractRooms in the future.
    """

    def __init__(
        self,
        env: Union[AbstractRooms, Wrapper],
        rooms_and_colors: Sequence[Tuple[str, str, str]],
        interact_action: int,
        seed: int,
    ):
        """Initialize.

        With these features, we assume the environment is using one
        color for all the space, exept few locations which
        have their own color.

        :param env: instance of AbstractRooms.
        :param rooms_and_colors: features of this class represent rooms.
            This argument is the association between a "room" and two colors.
            One for being ouside of the room, one inside.
            There should be a connection between the two colors.
            Example ("1", "r", "y"), where "1" is the name of a room,
            and the other two are colors.
        :param interact_action: One of the action is assumed to be an interaction.
        """
        self._env: AbstractRooms = env.unwrapped
        self._interact = interact_action
        self._seed = seed

        # Check associations and collect map
        self._common_color = " "
        self._colors2locations: Dict[str, str] = {}
        self._location2room: Dict[str, str] = {}
        for room, out_color, in_color in rooms_and_colors:
            assert out_color in self._env.rooms, "Every color should also in connections"
            assert in_color in self._env.rooms, "Every color should also in connections"
            self._colors2locations[out_color] = "out" + room
            self._colors2locations[in_color] = "in" + room
            self._location2room["out" + room] = room
            self._location2room["in" + room] = room

        # There should be one common room
        assert self._common_color is not None, (
            "There should be at least one pair with location 'none'"
        )
        # Are connections reflecting int room/out room/common room?
        for room, out_color, in_color in rooms_and_colors:
            assert (out_color, in_color) in self._env.rooms_connections, f"{out_color, in_color}"
            assert (self._common_color, out_color) in self._env.rooms_connections, f"{self._common_color, out_color}"
            assert (self._common_color, in_color) not in self._env.rooms_connections, f"{self._common_color, in_color}"
            for room2, out_color2, in_color2 in rooms_and_colors:
                if room != room2:
                    assert (out_color, out_color2) not in self._env.rooms_connections, f"{out_color}, {out_color2}"
                    assert (out_color, in_color2) not in self._env.rooms_connections, f"{out_color}, {in_color2}"
                    assert (in_color, in_color2) not in self._env.rooms_connections, f"{in_color}, {in_color2}"

        # Store fluents
        self.fluents: Set[str] = set()
        self.fluents.update(self._colors2locations.values())
        self.fluents.add("person")  # A person was detected
        self.fluents.add("closed")  # The door is closed
        self.fluents.add("bip")     # Just executed the interact action

        logger.debug(f"Office Fluents, colors2locations: {self._colors2locations}")
        logger.debug(f"Office Fluents, fluents: {self.fluents}")

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def on_episode_start(self):
        """Call when episode starts.

        Some features are not present in AbstractRooms;
        sampling them here.
        """
        # For each room
        self._episode_doors: Dict[str, bool] = {}
        self._episode_persons: Dict[str, bool] = {}
        for room in self._location2room.values():
            self._episode_doors[room] = random.random() > 0.5
            self._episode_persons[room] = random.random() > 0.5

    def __call__(self, obs: int, action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractRooms` environment. Rooms are interpreted as
            locations here.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        fluents = set()

        # Where the agent is
        color = self._env._id2room[obs]
        if color != self._common_color:
            location = self._colors2locations[color]
            fluents.add(location)

            # What it perceives
            room = self._location2room[location]
            if location.startswith("out") and self._episode_doors[room]:
                fluents.add("closed")
            elif location.startswith("in") and self._episode_persons[room]:
                fluents.add("person")

        # Whether it had interacted
        if action == self._interact:
            fluents.add("bip")

        return fluents


def make(params: Dict[str, Any], log_dir: Optional[str] = None):
    """Make the sapientino abstract state environment (agent teleports).

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: an object that respects the gym.Env interface.
    """
    # Check
    if params["shaping"]:
        raise ValueError("Can't shape rewards in the most abstract environment.")

    # Base env
    env = AbstractRooms(
        rooms_connectivity=params["rooms_connectivity"],
        initial_room=params["initial_room"],
        p_failure=params["fail_p"],
    )

    # Fluents for this environment
    fluent_extractor: FluentExtractor
    if params["fluents"] == "rooms":
        fluent_extractor = AbstractRoomsFluents(env)
    elif params["fluents"] == "party":
        env = WithExtraAction(env)
        fluent_extractor = AbstractPartyFluents(
            env=env,
            rooms_and_locations=params["rooms_and_locations"],
            interact_action=env.action_space.n - 1,
        )
    elif params["fluents"] == "office":
        env = WithExtraAction(env)
        fluent_extractor = AbstractOfficeFluents(
            env=env,
            rooms_and_colors=params["rooms_and_colors"],
            interact_action=env.action_space.n - 1,
            seed=params["seed"],
        )
    else:
        raise ValueError(params["fluents"])

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
    )
    env = FlattenAutomataStates(env)

    # Reward shift
    if params["reward_shift"] != 0:
        env = RewardShift(env, params["reward_shift"])

    # Time limit
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    return env
