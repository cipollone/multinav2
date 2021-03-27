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

"""Definition of non-Markovian goals for the environments.

Since the same goal can be used for different environments, we define them
here, so that these are shared.
"""

import pickle
from typing import Optional, Sequence

from gym.spaces import Discrete
from pythomata.impl.simple import SimpleDFA
from pythomata.utils import powerset
from temprl.wrapper import TemporalGoal

from multinav.envs.base import AbstractFluents


class SapientinoGoal(TemporalGoal):
    """Temporal goals for sapientino environments.

    This class defines temporal goals for all the sapientino environments.
    The goal is to visit all the given colors (these are positions in
    the environments) in some fixed order.

    Rigth not, just for efficiency, the automaton is defined directly and not
    built from a temporal formula.
    """

    def __init__(
        self,
        colors: Sequence[str],
        fluents: AbstractFluents,
        reward: Optional[float] = 1.0,
        save_to: Optional[str] = None,
    ):
        """Initialize.

        :param colors: a sequence of colors, these are the positions that
            the agent must reach with the correct order.
        :param fluents: a fluents evaluator. All colors must be fluents, so
            that we know when the agent is in each position.
        :param reward: reward suplied when reward is reached.
        :param save_to: path where the automaton should be exported.
        """
        # Check
        if not all((color in fluents.fluents for color in colors)):
            raise ValueError("Some color has no associated fluent to evaluate it")

        # Make automaton for this sequence
        automaton = self._make_sapientino_automaton(colors)

        # Super
        TemporalGoal.__init__(
            self,
            formula=None,  # Provinding automaton directly
            reward=reward,
            automaton=automaton,
            labels=set(colors),
            extract_fluents=fluents.evaluate,
            reward_shaping=False,
            zero_terminal_state=False,
        )

        # Maybe save
        if save_to:
            self.automaton.to_graphviz().render(save_to)

    @staticmethod
    def _make_sapientino_automaton(colors: Sequence[str]) -> SimpleDFA:
        """Make the automaton from a sequence of colors."""
        alphabet = set(map(frozenset, powerset(set(colors))))

        nb_states = len(colors) + 2
        initial_state = 0
        current_state = initial_state
        sink = nb_states - 1
        accepting = nb_states - 2
        states = {initial_state, sink}
        transitions = {}
        for c in colors:
            next_state = current_state + 1
            for symbol in alphabet:
                if c in symbol:
                    transitions.setdefault(current_state, {})[symbol] = next_state
                else:
                    transitions.setdefault(current_state, {})[symbol] = sink
            current_state = next_state
            states.add(current_state)

        for symbol in alphabet:
            transitions.setdefault(current_state, {})[symbol] = sink
            transitions.setdefault(sink, {})[symbol] = sink

        dfa = SimpleDFA(states, alphabet, initial_state, {accepting}, transitions)
        return dfa.trim().complete()

    @property
    def observation_space(self) -> Discrete:
        """Return the observation space.

        NOTE: Temprl returns automata states+1, we don't want that
        if we already have a complete automaton.
        """
        return Discrete(len(self._automaton.states))


class SapientinoOfficeGoal(TemporalGoal):
    """Define a temporal goal of a navigation task.

    The goal of the agent is: to reach room A, if the door is open,
    enter room A; when inside, if a person is detected, call that person with
    the "visit" action. After this, go to room B and go on.
    """

    def __init__(
        self,
        n_rooms: int,
        fluents: AbstractFluents,
        saved_automaton: str,
        reward: float,
        save_to: Optional[str] = None,
    ):
        """Initialize.

        :param n_rooms: the number of rooms in this environment. From this
            will follow the fluents and automaton states.
        :param fluents: this object contains the fluents valuation function.
            This method will check that the valuated fluents are the expected
            ones.
        :param saved_automaton: path to a saved automaton corresponding
            to a temporal goal. Fluents must match.
        :param reward: reward suplied when reward is reached.
        :param save_to: path where the automaton should be exported.
        """
        # Define the propositional symbols
        expected_fluents = {"bip", "person", "closed"}
        for i in range(1, n_rooms + 1):
            at_room, in_room = f"at{i}", f"in{i}"
            expected_fluents.add(at_room)
            expected_fluents.add(in_room)

        # Load automaton
        with open(saved_automaton, "rb") as f:
            automaton: SimpleDFA = pickle.load(f)

        # Check same fluents in valuations
        if not fluents.fluents == expected_fluents:
            raise ValueError(
                f"Symbols do not match: {fluents.fluents} != {expected_fluents}"
            )
        # NOTE: not checking also for automaton. Assuming correct

        # Super
        TemporalGoal.__init__(
            self,
            formula=None,  # Provinding automaton directly
            reward=reward,
            automaton=automaton,
            labels=fluents,
            extract_fluents=fluents.evaluate,
            reward_shaping=False,
            zero_terminal_state=False,
        )

        # Maybe save
        if save_to:
            self.automaton.to_graphviz().render(save_to)

    @property
    def observation_space(self) -> Discrete:
        """Return the observation space.

        NOTE: Temprl returns automata states+1, we don't want that
        if we already have a complete automaton.
        """
        return Discrete(len(self._automaton.states))
