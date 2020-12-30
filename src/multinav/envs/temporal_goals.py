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

from typing import Optional, Sequence

from flloat.semantics import PLInterpretation
from pythomata.base import TransitionFunction
from pythomata.dfa import DFA
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
    ):
        """Initialize.

        :param colors: a sequence of colors, these are the positions that
            the agent must reach with the correct order.
        :param fluents: a fluents evaluator. All colors must be fluents, so
            that we know when the agent is in each position.
        :param reward: reward suplied when reward is reached.
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

    @staticmethod
    def _make_sapientino_automaton(colors: Sequence[str]) -> DFA:
        """Make the automaton from a sequence of colors."""
        alphabet = set(map(PLInterpretation, powerset(set(colors))))
        false_ = PLInterpretation(set())

        nb_states = len(colors) + 2
        initial_state = 0
        current_state = initial_state
        sink = nb_states - 1
        accepting = nb_states - 2
        states = {initial_state, sink}
        transitions: TransitionFunction = {}
        for c in colors:
            next_state = current_state + 1
            for symbol in alphabet:
                if c in symbol.true_propositions:
                    transitions.setdefault(current_state, {})[symbol] = next_state
                else:
                    transitions.setdefault(current_state, {})[symbol] = sink
                transitions.setdefault(current_state, {})[false_] = current_state
            current_state = next_state
            states.add(current_state)

        for symbol in alphabet:
            transitions.setdefault(current_state, {})[symbol] = current_state
            transitions.setdefault(sink, {})[symbol] = sink

        dfa = DFA(states, alphabet, initial_state, {accepting}, transitions)
        return dfa.trim().complete()
