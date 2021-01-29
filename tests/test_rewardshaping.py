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

"""Tests for Reward shapers."""

import numpy as np

from multinav.envs.temporal_goals import SapientinoGoal
from multinav.helpers.reward_shaping import (
    AutomatonRS,
    PotentialRewardShaper,
    ValueFunctionRS,
)


def test_zeroterminalstates():
    """Test zero_terminal_state."""
    shaper = PotentialRewardShaper(
        potential_function=lambda x: x * x,
        gamma=0.9,
        zero_terminal_state=True,
    )

    observations = [7, 9, 2, 2, 7, 8, 5, 1, 8, 5]
    potential_0 = 49

    discounted_return = 0
    shaper.reset(observations[0])
    observations = observations[1:]
    for i, obs in enumerate(observations):
        reward = shaper.step(obs, 0.0, i == len(observations) - 1)
        discounted_return += (0.9 ** i) * reward

    assert np.isclose(discounted_return, -potential_0)


def test_reward_shaper():
    """Test ValueFunctionRS."""
    gamma = 0.5

    shaper = ValueFunctionRS(
        value_function=lambda x: x,
        mapping_function=lambda x: x + 1,
        gamma=gamma,
        zero_terminal_state=True,
    )

    shaper.reset(-1)
    assert shaper.step(-1, 0.0, False) == 0
    assert shaper.step(0, 0.0, False) == 0.5
    assert shaper.step(2, 0.0, False) == 0.5
    assert shaper.step(2, 0.0, False) == -1.5
    assert shaper.step(2, 0.0, True) == -3


def test_null_shaper():
    """Test null reward shaping."""
    shaper = ValueFunctionRS(
        value_function=lambda _: 0.0,
        mapping_function=lambda x: x,
        gamma=0.9,
        zero_terminal_state=False,
    )

    shaper.reset(-1)
    assert shaper.step(-1, 0.0, False) == 0.0
    assert shaper.step(2, 0.0, False) == 0.0
    assert shaper.step(3, 0.0, False) == 0.0


def test_automatonrs():
    """Test reward shaping from DFA."""

    class Fluents:
        fluents = {"red", "blue"}

        def evaluate(self, obs, action):
            return None

    tg = SapientinoGoal(
        colors=["red", "blue"],
        fluents=Fluents(),
        reward=2.0,
        save_to=None,
    )

    shaper = AutomatonRS(
        goal=tg.automaton,
        gamma=1.0,
        rescale=False,
        cancel_reward=False,
    )

    # tf.automaton sink state is N 2
    shaper.reset(("", [0]))
    assert shaper.step(("", [0]), 0.0, False) == 0.0
    assert shaper.step(("", [0]), 0.0, False) == 0.0
    assert shaper.step(("", [1]), 0.0, False) == 1.0
    assert shaper.step(("", [3]), 0.0, False) == 1.0
    assert shaper.step(("", [3]), 0.0, False) == 0.0
    assert shaper.step(("", [2]), 0.0, False) == -3.0

    shaper = AutomatonRS(
        goal=tg.automaton,
        gamma=1.0,
        rescale=True,
        cancel_reward=True,
    )

    shaper.reset(("", [0]))
    shaper.step(("", [1]), 0.0, False)
    assert np.isclose(shaper.step(("", [3]), 2.0, True), -1.6666, atol=1e-3)
