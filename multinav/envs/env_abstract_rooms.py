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
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np
from gym.wrappers import TimeLimit
from gym_sapientino.core.types import id2color as room2color
from temprl.types import Interpretation

from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.gym import MyDiscreteEnv, Transitions
from multinav.wrappers.reward_shaping import RewardShift
from multinav.wrappers.temprl import FlattenAutomataStates

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

    def __init__(self, env: AbstractRooms, interact_action: int):
        """Initialize.

        :param env: instance of AbstractRooms.
        """
        self._env = env
        rooms_and_locations = [
            ("r", "none"),
            ("g", "bar"),
            ("b", "alice"),
            ("y", "carol"),
        ]
        self._rooms2locations = dict(rooms_and_locations)
        self._interact = interact_action
        self.fluents = {"at_" + loc for loc in self._rooms2locations.values()}

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
            room = self._env._id2room[obs]
            location = self._rooms2locations[room]
            if location != "none":
                fluents.add("at_" + location)

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
        fluent_extractor = AbstractPartyFluents(env, interact_action=env.action_space.n - 1)
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
