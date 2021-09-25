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
from gym.wrappers import TimeLimit
from PIL import Image
from temprl.types import Action, FluentExtractor, Interpretation

from multinav.envs import sapientino_defs, temporal_goals
from multinav.helpers.general import classproperty
from multinav.helpers.gym import (
    MyDiscreteEnv,
    Probability,
    State,
    Transitions,
    from_discrete_env_to_graphviz,
)
from multinav.wrappers.temprl import FlattenAutomataStates
from multinav.wrappers.utils import CompleteActions


class AbstractSapientino(MyDiscreteEnv):
    """Abstract Sapientino environment."""

    def __init__(self, nb_colors: int, failure_probability: float = 0.1):
        """
        Initialize the environment.

        :param nb_colors: the number of color to consider.
        """
        self._nb_colors = nb_colors
        self._failure_probability = failure_probability
        model = self._make_transitions()
        isd = np.zeros(self.nb_states)
        isd[self.initial_state] = 1.0
        super().__init__(self.nb_states, self.nb_actions, model, isd)

    @property
    def nb_colors(self) -> int:
        """Get the number of colors."""
        return self._nb_colors

    @property
    def fail_prob(self) -> Probability:
        """Get the failure probability."""
        return self._failure_probability

    @property
    def initial_state(self) -> int:
        """Get the initial state."""
        return 0

    @property
    def nb_states(self) -> int:
        """
        Get the number of states.

        That is:
        - one state for the corridor
        - one state for each color
        NOTE: Some parts of this module assume that the corridor, or "blank"
        state is number 0. This is also the same convention of gym-sapientino.

        :return: the number of states.
        """
        return self.nb_colors + 1

    @property
    def nb_actions(self) -> int:
        """
        Get the number of actions.

        It includes:
        - go to to each color (+nb_colors)
        - go back to the corridor (+1)
        - visit color (+1)

        :return: the number of actions.
        """
        return self.nb_colors + 1 + 1

    def _is_legal_color(self, color_id: int):
        """Check that it is a legal color."""
        assert 0 <= color_id < self.nb_colors, f"{color_id} is not a legal color."

    def state_from_color(self, color_id: int):
        """
        Get the state from the color.

        Sum +1 because 0 is the initial state.
        """
        self._is_legal_color(color_id)
        return color_id + 1

    def action_goto_color_from_color(self, color_id: int) -> Action:
        """Get the action "goto color" from the color id."""
        assert 0 <= color_id < self.nb_colors, f"{color_id} is not a legal color."
        return color_id + 2

    @classproperty
    def goto_corridor(cls) -> Action:  # pylint: disable=no-self-argument
        """Get the action "goto corridor"."""
        return 0

    @classproperty
    def visit_color(cls) -> Action:  # pylint: disable=no-self-argument
        """Get the action "visit_color"."""
        return 1

    def _make_transitions(self) -> Transitions:
        """
        Make the model.

        :return: the transitions.
        """
        model: Transitions = {}

        for color_id in range(self.nb_colors):
            color_state = self.state_from_color(color_id)

            # from the corridor, you can go to any color
            goto_color_action = self.action_goto_color_from_color(color_id)
            new_transition = (1.0 - self.fail_prob, color_state, 0.0, False)
            fail_transition = (self.fail_prob, self.initial_state, 0.0, False)
            model.setdefault(self.initial_state, {}).setdefault(
                goto_color_action, []
            ).extend([new_transition, fail_transition])

            # if you visit a color, you remain in the same state.
            # NOTE: these two transitions are equivalent to a single transition
            #   it may be useful later, though.
            new_transition = (1.0 - self.fail_prob, color_state, 0.0, False)
            fail_transition = (self.fail_prob, color_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(self.visit_color, []).extend(
                [new_transition, fail_transition]
            )

            # from any color, you can go back to the corridor.
            new_transition = (1.0 - self.fail_prob, self.initial_state, 0.0, False)
            fail_transition = (self.fail_prob, color_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(self.goto_corridor, []).extend(
                [new_transition, fail_transition]
            )

            # TODO: experiments in AbstractSapientinoOffice require the
            # possibility to travel between pairs of colors (out and in of a
            # room). Adapted graph for this purpose.
            related_color = color_id + 1 if color_id % 2 == 0 else color_id - 1
            related_state = self.state_from_color(related_color)
            new_transition = (1.0 - self.fail_prob, related_state, 0.0, False)
            fail_transition = (self.fail_prob, color_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(
                self.action_goto_color_from_color(related_color), []
            ).extend([new_transition, fail_transition])

        return model

    def _action_to_string(self, action: Action):
        """From action to string."""
        self._is_legal_action(action)
        if action == self.goto_corridor:
            return "goto_corridor"
        if action == self.visit_color:
            return "visit"
        return f"goto_{action}"

    def _state_to_string(self, state: State):
        """From state to string.

        All sapientino environments interpret IDs as the same color.
        """
        self._is_legal_state(state)
        if state == 0:
            return "corridor"
        else:
            return sapientino_defs.int2color[state + 1]

    def render(self, mode="human"):
        """Render the environment (only rgb_array mode)."""
        assert mode == "rgb_array"
        graph = from_discrete_env_to_graphviz(
            self, state2str=self._state_to_string, action2str=self._action_to_string
        )
        image_file = io.BytesIO(graph.pipe(format="png"))
        array = np.array(Image.open(image_file))
        return array


class Fluents:
    """A fluent extractor for abstract sapientino."""

    def __init__(self, nb_colors: int):
        """Initialize.

        :param nb_colors: The number of colors/rooms in the environment.
        """
        base_id = 2  # the firsts are for corridor and wall
        self.fluents = {
            sapientino_defs.int2color[i] for i in range(base_id, base_id + nb_colors)
        }

    def __call__(self, obs: int, action: int) -> Interpretation:
        """Respects temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation comes from an
            `AbstractSapientino` environment.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        if action == AbstractSapientino.visit_color:
            fluents = {sapientino_defs.int2color[obs]}
            if obs == 0:  # blank/corridor
                fluents = set()
        else:
            fluents = set()
        return {f for f in self.fluents if f in fluents}


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
