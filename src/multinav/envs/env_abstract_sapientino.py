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
from typing import Dict, List, Optional

import numpy as np
from gym.spaces import MultiDiscrete
from PIL import Image
from pythomata.dfa import DFA

from multinav.envs.base import AbstractFluents
from multinav.helpers.gym import (
    Action,
    MyDiscreteEnv,
    Probability,
    State,
    Transitions,
    from_discrete_env_to_graphviz,
)
from multinav.restraining_bolts.base import SapientinoRB
from multinav.wrappers.temprl import MyTemporalGoalWrapper


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
        ids = np.zeros(self.nb_states)
        ids[self.initial_state] = 1.0
        super().__init__(self.nb_states, self.nb_actions, model, ids)

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
        - one state for each color
        - one state for the corridor

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

    @property
    def goto_corridor(self) -> Action:
        """Get the action "goto corridor"."""
        return 0

    @property
    def visit_color(self) -> Action:
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
            new_transition = (1.0 - self.fail_prob, color_state, 0.0, False)
            fail_transition = (self.fail_prob, color_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(self.visit_color, []).extend(
                [new_transition, fail_transition]
            )

            # from any color, you can go back to the corridor.
            new_transition = (1.0, self.initial_state, 0.0, False)
            fail_transition = (self.fail_prob, color_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(self.goto_corridor, []).extend(
                [new_transition, fail_transition]
            )

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
        """From state to string.""" ""
        # TODO add color strings.
        self._is_legal_state(state)
        return str(state)

    def render(self, mode="human"):
        """Render the environment (only rgb_array mode)."""
        assert mode == "rgb_array"
        graph = from_discrete_env_to_graphviz(
            self, state2str=self._state_to_string, action2str=self._action_to_string
        )
        image_file = io.BytesIO(graph.pipe(format="png"))
        array = np.array(Image.open(image_file))
        return array


class AbstractSapientinoTemporalGoal(MyTemporalGoalWrapper, MyDiscreteEnv):
    """
    Abstract Sapientino with Temporal Goals.

    We need this instead of using the temporal goal wrapper because we
    need to build an explicit model.
    """

    def __init__(
        self,
        restraining_bolt: SapientinoRB,
        sapientino_args: Optional[List],
        sapientino_kwargs: Optional[Dict] = None,
    ):
        """Initialize the environment."""
        sapientino_args = sapientino_args or []
        sapientino_kwargs = sapientino_kwargs or {}
        self.unwrapped_env = AbstractSapientino(*sapientino_args, **sapientino_kwargs)
        self.temporal_goal = restraining_bolt.make_sapientino_goal()
        self.restraining_bolt = restraining_bolt
        MyTemporalGoalWrapper.__init__(self, self.unwrapped_env, [self.temporal_goal])  # type: ignore

        # flatten the observation space
        observation_space = MultiDiscrete(
            [
                self.unwrapped_env.observation_space.n,
                self.temporal_goal.observation_space.n,
            ]
        )

        # compute model
        model = self._compute_model()
        nb_states = observation_space.nvec.prod()
        nb_actions = self.unwrapped_env.nb_actions
        ids = np.zeros(nb_states)
        ids[0] = 1.0
        MyDiscreteEnv.__init__(self, nb_states, nb_actions, model, ids)
        self.observation_space = observation_space

    def _compute_model(self):
        """Compute the model."""
        model: Transitions = {}
        automaton: DFA = self.temporal_goal.automaton
        failure_state = len(automaton.states) - 1
        for automaton_state in automaton.states:
            done = automaton_state in automaton.accepting_states
            reward = 1.0 if done else 0.0
            for color_id in range(self.unwrapped_env.nb_colors):
                initial_state = (self.unwrapped_env.initial_state, automaton_state)
                color = self.unwrapped_env.state_from_color(color_id)
                color_state = (color, automaton_state)

                # from the corridor, you can go to any color
                goto_color_action = self.unwrapped_env.action_goto_color_from_color(
                    color_id
                )
                new_transition = (
                    1.0 - self.unwrapped_env.fail_prob,
                    color_state,
                    reward,
                    done,
                )
                fail_transition = (
                    self.unwrapped_env.fail_prob,
                    initial_state,
                    reward,
                    done,
                )
                model.setdefault(initial_state, {}).setdefault(
                    goto_color_action, []
                ).extend([new_transition, fail_transition])

                # if you visit a color, check the transition on the automaton.
                fluents = self.temporal_goal.extract_fluents(
                    color, self.unwrapped_env.visit_color
                )
                next_automaton_state = automaton.transition_function.get(
                    automaton_state, {}
                ).get(fluents, failure_state)
                next_state = (color, next_automaton_state)
                new_transition = (
                    1.0 - self.unwrapped_env.fail_prob,
                    next_state,
                    reward,
                    done,
                )
                fail_transition = (
                    self.unwrapped_env.fail_prob,
                    color_state,
                    reward,
                    done,
                )
                model.setdefault(color_state, {}).setdefault(
                    self.unwrapped_env.visit_color, []
                ).extend([new_transition, fail_transition])

                # from any color, you can go back to the corridor.
                new_transition = (1.0, initial_state, reward, done)
                fail_transition = (
                    self.unwrapped_env.fail_prob,
                    color_state,
                    reward,
                    done,
                )
                model.setdefault(color_state, {}).setdefault(
                    self.unwrapped_env.goto_corridor, []
                ).extend([new_transition, fail_transition])

        return model

    def reset(self):
        """Reset the environment."""
        obs, automata_states = super().reset()
        return tuple([obs] + automata_states)

    def step(self, a):
        """Do a step in the environment."""
        state, reward, done, info = super().step(a)
        obs, automata_states = state
        if self.temporal_goal.is_true():
            reward += self.temporal_goal.reward
            done = True
        new_state = tuple([obs] + automata_states)
        return new_state, reward, done, info

    def available_actions(self, state):
        """Get the available action from a state."""
        actions = set()
        for action, _transitions in self.P.get(state, {}).items():
            actions.add(action)
        return actions


class Fluents(AbstractFluents):
    """Define the propositions in this specific environment."""

    def __init__(self, colors_set: Set[str]):
        """Initialize.

        :param colors_set: a set of colors among the ones used by sapientino;
            this will be the set of fluents to evaluate.
        """
        # TODO:
        pass

    def evaluate(self, obs: Dict[str, float], action) -> PLInterpretation:
        """Missing."""
        # TODO: docstring and fn
        return None
