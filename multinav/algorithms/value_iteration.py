# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#
"""Value iteration implementation."""
from collections import defaultdict
from typing import Dict

import numpy as np

from multinav.helpers.gym import MyDiscreteEnv, iter_space


class _ValueIteration:
    """Execute value iteration."""

    def __init__(
        self,
        env: MyDiscreteEnv,
        max_iterations: int = 100,
        eps: float = 1e-8,
        discount: float = 0.9,
    ):
        self.env = env
        self.eps = eps
        self.discount = discount
        self.v = self._new_value_function()
        self.max_iterations = max_iterations
        self.policy: Dict = {}

    def _random(self):
        """Random initializer."""
        return np.random.random() * 1e-5

    def _new_value_function(self):
        """Reset value function."""
        return defaultdict(self._random)

    def __call__(self):
        """Run value iteration against a DiscreteEnv environment."""
        delta = np.inf
        iteration = 0
        while not delta < self.eps and iteration < self.max_iterations:
            delta = 0
            next_v = self._new_value_function()
            for s in self.env.P:
                vs = self.v[s]
                next_values = self._get_next_values(s)
                new_vs = max(next_values)
                next_v[s] = new_vs
                delta = max(delta, abs(vs - new_vs))
            self.v = next_v
            iteration += 1

        self._compute_optimal_policy()
        return self.v, self.policy

    def _get_next_values(self, state):
        """Get the next value, given state and action."""
        next_values = [
            sum(
                [
                    p * (r + self.discount * self.v[sp])
                    for (p, sp, r, _done) in self.env.P[state][action]
                ]
            )
            for action in iter_space(self.env.action_space)
            if action in self.env.available_actions(state)
        ]
        return next_values

    def _compute_optimal_policy(self):
        """Compute optimal policy from value function."""
        for s in self.env.P:
            action_values = self._get_next_values(s)
            new_action = np.argmax(action_values) if len(action_values) > 0 else None
            self.policy[s] = new_action


def value_iteration(*args, **kwargs):
    """Run value iteration."""
    return _ValueIteration(*args, **kwargs)()


def pretty_print_v(valuefn: dict):
    """Nicely print value function (just for debugging)."""
    print("Value function")
    states = list(valuefn.keys())
    states.sort(key=lambda k: k if isinstance(k, int) else 100)

    for state in states:
        print("  ", state, ": ", np.around(valuefn[state], decimals=2), sep="")
