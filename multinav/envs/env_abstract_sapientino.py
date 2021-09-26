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

import io
from typing import Any, Dict, Optional

import numpy as np
from gym.wrappers import TimeLimit
from PIL import Image
from temprl.types import Action, Interpretation

from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.gym import (
    MyDiscreteEnv,
    State,
    Transitions,
    from_discrete_env_to_graphviz,
)
from multinav.wrappers.temprl import FlattenAutomataStates
from multinav.wrappers.utils import Debugger


class AbstractSapientino(MyDiscreteEnv):
    """Abstract Sapientino environment."""

    def __init__(
        self,
        n_locations: int,
        p_failure: float = 0.0,
        initial_location: int = 0,
    ):
        """Initialize the environment.

        :param n_locations: the number of locations/states.
        :param p_failure: probability of failing a trainsition and remaining on
            the same state.
        :param initial_location: the initial state
        """
        self.n_locations = n_locations
        self.p_failure = p_failure
        model = self._make_transitions()
        isd = np.zeros(self.n_locations)
        self.initial_state = initial_location
        isd[self.initial_state] = 1.0
        super().__init__(self.nb_states, self.nb_actions, model, isd)

    @property
    def nb_states(self) -> int:
        """Get the number of states."""
        return self.n_locations

    @property
    def nb_actions(self) -> int:
        """Get the number of actions.

        One go_to action for each state + one interact action
        (aliases: visit, bip)
        """
        return self.n_locations + 1

    @property
    def action_interact(self) -> int:
        """Return the special "interact" action."""
        return self.nb_actions - 1

    def action_goto_state(self, state: int) -> int:
        """Return the Go-To action associated to a location."""
        assert self._is_state(state)
        return state

    def _is_state(self, state: int):
        """Check that it is a legal state."""
        return 0 <= state < self.nb_states

    def _is_action(self, action: int):
        """Check if tat is a legal action."""
        return 0 <= action < self.nb_actions

    def _make_transitions(self) -> Transitions:
        """Make the trainsition model."""
        model: Transitions = {}

        for from_location in range(self.n_locations):

            # You can go to any other location
            for to_location in range(self.n_locations):
                goto_action = self.action_goto_state(to_location)
                ok_transition = (1.0 - self.p_failure, to_location, 0.0, False)
                fail_transition = (self.p_failure, from_location, 0.0, False)
                model.setdefault(from_location, {}).setdefault(
                    goto_action, []).extend([ok_transition, fail_transition])

            # You can start the interaction
            ok_transition = (1.0, from_location, 0.0, False)
            model.setdefault(from_location, {}).setdefault(
                self.action_interact, []).extend([ok_transition])

        return model

    def _action_to_string(self, action: Action):
        """From action to string."""
        assert self._is_action(action)
        if action == self.action_interact:
            return "interact"
        else:
            return f"goto_{action}"

    def _state_to_string(self, state: State):
        """From state to string."""
        assert self._is_action(state)
        return f"location_{state}"

    def render(self, mode="human"):
        """Render the environment (only rgb_array mode)."""
        assert mode == "rgb_array"
        graph = from_discrete_env_to_graphviz(
            self, state2str=self._state_to_string, action2str=self._action_to_string
        )
        image_file = io.BytesIO(graph.pipe(format="png"))
        array = np.array(Image.open(image_file))
        return array


class OfficeAbstractSapientino(AbstractSapientino):
    """AbstractSapientino with office environment.

    This assigns a role to each location: either the corridor, or a location
    close to a door, or the inside of a room. It simulates the presence of
    doors and people. The observation is the same as in AbstractSapientino,
    the additional features, computed after each reset or step, are stored in
    self.obs_features.
    """

    def __init__(self, n_rooms: int, p_failure: float, seed: int):
        """Initialize.

        :param n_rooms: number of rooms. Each room has two associated
            locations. Also there's one that is not related to any room.
        :param p_failure: probability of failing a go-to action.
        """
        # Instantiate
        self.n_rooms = n_rooms
        n_locations = n_rooms * 2 + 1
        super().__init__(
            n_locations=n_locations,
            p_failure=p_failure,
            initial_location=n_locations - 1,
        )

        # Translation
        self.location2name = (
            [f"out{i}" for i in range(n_rooms)]
            + [f"in{i}" for i in range(n_rooms)] + ["corridor"]
        )
        self.location2room = (
            [i for i in range(n_rooms)]
            + [i for i in range(n_rooms)] + [-1]
        )

        self.__rng = np.random.default_rng(seed)

    def _update_features(self, observation) -> None:
        """Compute a dictionary of features."""
        name = self.location2name[observation]
        room = self.location2room[observation]
        self.obs_features = dict(
            location=name,
            person=(name != "corridor" and self._person_in[room] == 1),
            closed=(name != "corridor" and self._door_closed[room] == 1),
        )

    def step(self, action: int):
        """Gym step."""
        observation, reward, done, info = super().step(action)
        self._update_features(observation)
        return observation, reward, done, info

    def reset(self):
        """Gym reset."""
        obs = super().reset()

        # Choose doors and persons
        self._door_closed = self.__rng.integers(0, 2, size=self.n_rooms)
        self._person_in = self.__rng.integers(0, 2, size=self.n_rooms)

        self._update_features(obs)

        return obs


class OfficeFluents(FluentExtractor):
    """Define propositions for OfficeAbstractSapientino."""

    def __init__(self, env: OfficeAbstractSapientino):
        """Initialize.

        :param env: instance of OfficeAbstractSapientino.
        """
        self._env = env

        # Define the propositional symbols
        self.fluents = {"interact", "person", "closed"}
        for i in range(self._env.n_rooms):
            self.fluents.add(f"out{i}")
            self.fluents.add(f"in{i}")

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def __call__(self, _obs: int, action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `OfficeAbstractSapientino` environment.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        fluents = set()
        if self._env.obs_features["person"]:
            fluents.add("person")
        if self._env.obs_features["closed"]:
            fluents.add("closed")
        if self._env.obs_features["location"] != "corridor":
            fluents.add(self._env.obs_features["location"])
        if action == self._env.action_interact:
            fluents.add("interact")
        assert fluents <= self.all
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
    env = OfficeAbstractSapientino(
        n_rooms=params["nb_rooms"],
        p_failure=params["sapientino_fail_p"],
        seed=params["seed"],
    )

    # Fluents for this environment
    fluent_extractor = OfficeFluents(env)

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
    )
    env = FlattenAutomataStates(env)

    # Time limit
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    return env
