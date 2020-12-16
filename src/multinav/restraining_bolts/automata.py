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

"""This module contains functions to manually build the goal automaton."""

from typing import Any, Callable, Sequence

from flloat.semantics import PLInterpretation
from pythomata.base import TransitionFunction
from pythomata.dfa import DFA
from pythomata.utils import powerset
from temprl.wrapper import TemporalGoal


def _make_automaton(colors: Sequence[str]) -> DFA:
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

    return DFA(states, alphabet, initial_state, {accepting}, transitions)


def make_sapientino_goal_with_automata(
    colors: Sequence[str],
    fluent_extractor: Callable[[Any, Any], PLInterpretation],
    reward: float = 10.0,
) -> TemporalGoal:
    """
    Make Sapientino goal, by building the automaton manually.

    :param colors: the sequence of colors to visit.
    :param fluent_extractor: the fluent extractor (a callable
      that accepts a state and an action and returns a set of fluents.
    :param reward: the reward.
    :return: None
    """
    dfa = _make_automaton(colors)
    return TemporalGoal(
        reward=reward,
        labels=set(colors),
        automaton=dfa,
        reward_shaping=False,
        zero_terminal_state=False,
        extract_fluents=fluent_extractor,
    )
