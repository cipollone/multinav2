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
from typing import Any, Dict, Optional, cast

import numpy as np
from gym import spaces
from gym.wrappers import TimeLimit
from PIL import Image
from temprl.types import Action, FluentExtractor, Interpretation

from multinav.envs import sapientino_defs, temporal_goals
from multinav.helpers.gym import (
    MyDiscreteEnv,
    State,
    Transitions,
    from_discrete_env_to_graphviz,
)
from multinav.wrappers.temprl import FlattenAutomataStates
from multinav.wrappers.utils import CompleteActions


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
    doors and people. The observation becomes a pair, where the first element
    is the orignal one, and the second stores a dictionarly with this
    additional infos.
    """

    def __init__(self, n_rooms: int, p_failure: float, seed: int):
        """Initialize.

        :param n_rooms: number of rooms. Each room has two associated
            locations. Also there's one that is not related to any room.
        :param p_failure: probability of failing a go-to action.
        """
        # Instantiate
        self._n_rooms = n_rooms
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

        # Observation NOTE: the space for the second element is wrong
        assert isinstance(self.observation_space, spaces.Discrete)
        self.observation_space = spaces.Tuple((
            self.observation_space, spaces.Dict(dict(
                unspecified_dict=spaces.Discrete(2)
            ))))

        self.__rng = np.random.default_rng(seed)

    def _get_features(self, observation) -> Dict[str, Any]:
        """Compute a dictionary of features."""
        name = self.location2name[observation]
        room = self.location2room[observation]
        return dict(
            location=name,
            person=(self._person_in[room] == 1 and name != "corridor"),
            closed=(self._door_closed[room] == 1 and name != "corridor"),
        )

    def step(self, action: int):
        """Gym step."""
        observation, reward, done, info = super().step(action)
        obs2 = self._get_features(observation)
        return (observation, obs2), reward, done, info

    def reset(self):
        """Gym reset."""
        obs = super().reset()

        # Choose doors and persons
        self._door_closed = self.__rng.integers(0, 2, size=self._n_rooms)
        self._person_in = self.__rng.integers(0, 2, size=self._n_rooms)

        return obs, self._get_features(obs)


# TODO: until here


class OfficeFluents:
    """Define propositions for AbstractSapientino with Office goal."""

    _colors_to_room = {
        "red": "at1",
        "green": "in1",
        "blue": "at2",
        "yellow": "in2",
        "pink": "at3",
        "brown": "in3",
        "gray": "at4",
        "orange": "in4",
    }

    def __init__(self, n_rooms: int, seed: int):
        """Initialize.

        :param nb_rooms: number of rooms to navigate.
        """
        assert n_rooms < 5, "Can't support more than four rooms"

        # Define the propositional symbols
        self.fluents = {"bip", "person", "closed"}
        for i in range(1, n_rooms + 1):
            at_room, in_room = f"at{i}", f"in{i}"
            self.fluents.add(at_room)
            self.fluents.add(in_room)

        self._rng = np.random.default_rng(seed)
        self._n_rooms = n_rooms

    def __call__(self, obs: int, action: int) -> Interpretation:
        """Respects temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractSapientino` environment.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        fluents = set()
        if obs != 0:
            color = sapientino_defs.int2color[obs + 1]
            fluents.add(self._colors_to_room[color])
        if action == AbstractSapientino.visit_color:
            fluents.add("bip")

        # Doors and people are not in the observation. Valuate at random
        samples = self._rng.integers(0, 2, size=2)
        if samples[0] == 1:
            fluents.add("closed")
        if samples[1] == 1:
            fluents.add("person")

        return {f for f in self.fluents if f in fluents}

    def evaluations_prob(self, obs, action):
        """Evaluate and return associated probabilities."""
        fluents = self(obs, action)
        fluents_dict = {f: f in fluents for f in self.fluents}
        fluents_dict["closed"] = False
        fluents_dict["person"] = False

        p = 1.0 / 4
        values = [(dict(fluents_dict), p) for _ in range(4)]
        values[1][0].update({"closed": True}), p
        values[2][0].update({"person": True}), p
        values[3][0].update({"closed": True, "person": True}), p
        return values


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
    env = AbstractSapientino(
        nb_colors=params["nb_rooms"] * 2,
        failure_probability=params["sapientino_fail_p"],
    )

    # Admit all actions
    env = CompleteActions(env)

    # Fluents for this environment
    fluent_extractor = OfficeFluents(
        n_rooms=params["nb_rooms"],
        seed=params["seed"],
    )

    # Apply temporal goals to this env
    env = temporal_goals.with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=cast(FluentExtractor, fluent_extractor),
        log_dir=log_dir,
    )
    env = FlattenAutomataStates(env)

    # Time limit
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    return env
