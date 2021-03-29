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
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image
from pythomata.impl.symbolic import PropositionalInterpretation

from multinav.envs import sapientino_defs
from multinav.envs.base import AbstractFluents
from multinav.envs.temporal_goals import SapientinoGoal, SapientinoOfficeGoal
from multinav.helpers.general import classproperty
from multinav.helpers.gym import (
    Action,
    MyDiscreteEnv,
    Probability,
    State,
    Transition,
    Transitions,
    from_discrete_env_to_graphviz,
    iter_space,
)
from multinav.wrappers.temprl import FlattenAutomataStates, MyTemporalGoalWrapper


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


class AbstractSapientinoTemporalGoal(MyDiscreteEnv):
    """
    Abstract Sapientino with Temporal Goals.

    We need this instead of using the temporal goal wrapper because we
    need to build an explicit model.
    """

    def __init__(
        self,
        *,
        tg_reward: float = 1.0,
        save_to: Optional[str] = None,
        **sapientino_kwargs,
    ):
        """Initialize the environment.

        :param tg_reward: reward supplied when the temporal goal is reached.
        :param save_to: path where the automaton temporal goal should be saved.
        """
        # Make AbstractSapientino
        self.sapientino_env = AbstractSapientino(**sapientino_kwargs)

        # Make a temporal goal
        nb_colors = sapientino_kwargs["nb_colors"]
        first_color = 2
        color_sequence = [
            sapientino_defs.int2color[i] for i in range(first_color, nb_colors + first_color)]
        self.fluents = Fluents(nb_colors=nb_colors)
        self.temporal_goal = SapientinoGoal(
            colors=color_sequence,
            fluents=self.fluents,
            reward=tg_reward,
            save_to=save_to,
        )

        # Build env with temporal goal
        self.temporal_env = FlattenAutomataStates(
            MyTemporalGoalWrapper(self.sapientino_env, [self.temporal_goal])
        )

        # compute model
        model: Transitions = {}
        initial_state = self.temporal_env.reset()  # because deterministic
        self._generate_transitions(model, initial_state)
        # Complete with unreachable states
        for state in iter_space(self.temporal_env.observation_space):
            model.setdefault(state, {})

        # Set discrete env
        nb_states = self.temporal_env.observation_space.nvec.prod()
        nb_actions = self.sapientino_env.nb_actions
        isd = np.zeros(nb_states)
        isd[0] = 1.0
        MyDiscreteEnv.__init__(self, nS=nb_states, nA=nb_actions, P=model, isd=isd)

        # Update observation space
        self.observation_space = self.temporal_env.observation_space

    def _generate_transitions(
        self,
        model: Transitions,
        state: State,
    ):
        """Recusively visit states and generate transitions."""
        if state in model:
            return

        # Transition function from state
        automaton = self.temporal_goal.automaton
        sapientino_tf = self.sapientino_env.P[state[0]]

        # For all actions
        model[state] = {}
        for action in sapientino_tf:
            new_transitions: List[Transition] = []
            model[state][action] = new_transitions

            # For all nondeterministic transitions
            for transition in sapientino_tf[action]:
                p, sap_state, sap_reward, _sap_done = transition

                # Possible evaluations
                fluents_list = self.fluents.evaluations_prob(sap_state, action)
                for fluents, fluents_p in fluents_list:

                    automaton_state = automaton.get_successor(state[1], fluents)

                    # Compose state
                    goal_reached = automaton_state in automaton.accepting_states
                    new_reward = sap_reward + (
                        0.0 if goal_reached else -1.0
                    )
                    new_done = goal_reached
                    new_state = (sap_state, automaton_state)

                    new_transitions.append(
                        (p * fluents_p, new_state, new_reward, new_done))

                    # Recurse
                    self._generate_transitions(model, new_state)

    def render(self, mode="human"):
        """Render with temporal goal; mode is ignored."""
        return self.temporal_env.render(mode="rgb_array")

    def reset(self):
        """Reset the environment."""
        return self.temporal_env.reset()

    def step(self, action):
        """Do a step on the environment."""
        return self.temporal_env.step(action)


class AbstractSapientinoOffice(AbstractSapientinoTemporalGoal):
    """AbstractSapientino with the office scenario temporal goal."""

    def __init__(
        self,
        *,
        nb_rooms: int,
        tg_reward: float,
        saved_automaton: str,
        save_to: Optional[str] = None,
        **sapientino_kwargs,
    ):
        """Initialize the environment.

        :param nb_rooms: number of rooms to visit and get inside.
        :param tg_reward: reward supplied when the temporal goal is reached.
        :param saved_automaton: path to a saved DFA that represents the
            temporal goal. Transitions must be interpretations of fluents.
        :param save_to: path where the automaton temporal goal should be saved.
        """
        # Define temporal goal
        self.fluents = OfficeFluents(n_rooms=nb_rooms)
        self.temporal_goal = SapientinoOfficeGoal(
            n_rooms=nb_rooms,
            fluents=self.fluents,
            saved_automaton=saved_automaton,
            reward=tg_reward,
            save_to=save_to,
        )

        # Make AbstractSapientino
        sapientino_kwargs["nb_colors"] = nb_rooms * 2
        self.sapientino_env = AbstractSapientino(**sapientino_kwargs)

        # Build env with temporal goal
        self.temporal_env = FlattenAutomataStates(
            MyTemporalGoalWrapper(self.sapientino_env, [self.temporal_goal])
        )

        # compute model
        model: Transitions = {}
        initial_state = self.temporal_env.reset()  # because deterministic
        self._generate_transitions(model, initial_state)
        # Complete with unreachable states
        for state in iter_space(self.temporal_env.observation_space):
            model.setdefault(state, {})

        # Set discrete env
        nb_states = self.temporal_env.observation_space.nvec.prod()
        nb_actions = self.sapientino_env.nb_actions
        isd = np.zeros(nb_states)
        isd[0] = 1.0
        MyDiscreteEnv.__init__(self, nS=nb_states, nA=nb_actions, P=model, isd=isd)

        # Update observation space
        self.observation_space = self.temporal_env.observation_space


class Fluents(AbstractFluents):
    """Define the propositions for `AbstractSapientino`."""

    def __init__(self, nb_colors: int):
        """Initialize.

        :param nb_colors: The number of colors/rooms in the environment.
        """
        base_id = 2  # the firsts are for corridor and wall
        self.fluents = {
            sapientino_defs.int2color[i] for i in range(base_id, base_id + nb_colors)
        }

    def evaluate(self, obs: int, action: int) -> PropositionalInterpretation:
        """Respects AbstractFluents.evaluate.

        :param obs: assuming that the observation comes from an
            `AbstractSapientino` environment.
        :param action: the last action.
        """
        if action == AbstractSapientino.visit_color:
            fluents = {sapientino_defs.int2color[obs]}
            if obs == 0:  # blank/corridor
                fluents = set()
        else:
            fluents = set()
        return {f: f in fluents for f in self.fluents}


class OfficeFluents(AbstractFluents):
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

    def __init__(self, n_rooms: int):
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

        self._rng = np.random.default_rng()
        self._n_rooms = n_rooms

    def evaluate(self, obs: int, action: int) -> Set[str]:
        """Respects.AbstractFluents.evaluate.

        :param obs: assuming that the observation comes from an
            `AbstractSapientino` environment.
        :param action: the last action.
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

        return {f: f in fluents for f in self.fluents}

    def evaluations_prob(self, obs, action):
        fluents = self.evaluate(obs, action)
        fluents["closed"] = False
        fluents["person"] = False

        p = 1.0 / 4
        values = [(dict(fluents), p) for i in range(4)]
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

    # Build
    env = AbstractSapientinoOffice(
        nb_rooms=params["nb_rooms"],
        tg_reward=params["tg_reward"],
        saved_automaton=params["tg_automaton"],
        save_to=os.path.join(log_dir, "reward-dfa.dot") if log_dir else None,
        failure_probability=params["sapientino_fail_p"],
    )
    return env
