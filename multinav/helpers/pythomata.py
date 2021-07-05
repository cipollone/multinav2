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
"""Helpers related to Pythomata."""
from typing import Any, Dict, Optional, Tuple

import graphviz
import sympy
from graphviz import Digraph
from pythomata.impl.simple import SimpleDFA
from sympy.logic.boolalg import Boolean, BooleanFalse


def to_graphviz_sym(  # noqa
    dfa: SimpleDFA,
    title: Optional[str] = None,
    states2colors: Optional[Dict[Any, str]] = None,
) -> Digraph:
    """To graphviz, symbolic."""
    symbols = max(dfa.alphabet, key=lambda x: len(x.true_propositions))
    states2colors = states2colors or {}
    g = graphviz.Digraph(format="svg")
    g.node("fake", style="invisible")
    for state in dfa.states:
        if state == dfa.initial_state:
            if state in dfa.accepting_states:
                g.node(str(state), root="true", shape="doublecircle")
            else:
                g.node(str(state), root="true")
        elif state in dfa.accepting_states:
            g.node(str(state), shape="doublecircle")
        else:
            g.node(str(state))

        if state in states2colors:
            g.node(str(state), fillcolor="lightsalmon", style="filled")

    g.edge("fake", str(dfa.initial_state), style="bold")
    for start in dfa.transition_function:
        symbolic_transitions: Dict[Tuple[Any, Any], Boolean] = {}
        for symbol, end in dfa.transition_function[start].items():
            tr = (start, end)
            if tr not in symbolic_transitions:
                symbolic_transitions[tr] = BooleanFalse()
            old = symbolic_transitions[tr]
            if len(symbol.true_propositions) > 0:
                formula = sympy.parse_expr(" & ".join(list(symbol.true_propositions)))
            else:
                formula = sympy.parse_expr(" & ".join(map(lambda s: "~" + s, symbols)))
            symbolic_transitions[tr] = old | formula

        for (start, end), formula in symbolic_transitions.items():
            formula = formula.simplify()
            g.edge(str(start), str(end), label=str(formula))

    if title:
        g.attr(label=title)
        g.attr(fontsize="20")

    return g


def to_graphviz(dfa: SimpleDFA, title: Optional[str] = None) -> Digraph:
    """Transform a pythomata.SimpleDFA into a graphviz.Digraph."""
    g = graphviz.Digraph(format="svg")
    g.node("fake", style="invisible")
    for state in dfa.states:
        if state == dfa.initial_state:
            if state in dfa.accepting_states:
                g.node(str(state), root="true", shape="doublecircle")
            else:
                g.node(str(state), root="true")
        elif state in dfa.accepting_states:
            g.node(str(state), shape="doublecircle")
        else:
            g.node(str(state))

    g.edge("fake", str(dfa.initial_state), style="bold")
    for start in dfa.transition_function:
        for symbol, end in dfa.transition_function[start].items():
            g.edge(str(start), str(end), label=str(symbol))

    if title:
        g.attr(label=title)
        g.attr(fontsize="20")

    return g
