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

"""Tests for Abstract Sapientino."""
from operator import itemgetter

import numpy as np
import pytest

from multinav.algorithms.value_iteration import value_iteration
from multinav.envs.env_abstract_sapientino import (
    AbstractSapientino,
    AbstractSapientinoTemporalGoal,
    Fluents,
)
from multinav.helpers.gym import Transition, Transitions


def test_abstract_sapientino():
    """Test abstract Sapientino."""
    env = AbstractSapientino(5, failure_probability=0.0)
    assert env.nA == 7
    assert env.nS == 6
    assert env.reset() == env.initial_state == 0

    with pytest.raises(ValueError, match="Cannot perform action 0 in state 0."):
        env.step(0)
    with pytest.raises(ValueError, match="Cannot perform action 1 in state 0."):
        env.step(1)

    # go to color 0
    next_state, reward, done, info = env.step(2)
    assert next_state == 1
    assert reward == 0.0
    assert not done

    # visit current color.
    next_state, reward, done, info = env.step(1)
    assert next_state == 1
    assert reward == 0.0
    assert not done

    for action in range(2, env.nA):
        with pytest.raises(
            ValueError, match=f"Cannot perform action {action} in state 1."
        ):
            env.step(action)

    # go back to corridor.
    next_state, reward, done, info = env.step(0)
    assert next_state == 0
    assert reward == 0.0
    assert not done


class _RewardWrapper(AbstractSapientino):
    """Add reward when visiting the last color."""

    def _make_transitions(self) -> Transitions:
        result = super()._make_transitions()
        target_color = self.nb_colors - 1
        target_action = self.visit_color

        state = self.state_from_color(target_color)
        transition = result[state][target_action][0]
        new_transition = list(transition)
        new_transition[2] = 1.0
        new_transition_tuple: Transition = tuple(new_transition)  # type: ignore
        result[state][target_action] = [new_transition_tuple]

        return result


def test_value_iteration():
    """Test value iteration on abstract Sapientino."""
    env = _RewardWrapper(5, failure_probability=0.0)
    v, policy = value_iteration(env, discount=0.9, max_iterations=200)
    actual_values = np.array(
        list(map(itemgetter(1), sorted(v.items(), key=lambda x: x[0])))
    )
    expected_values = np.array([9, 8.1, 8.1, 8.1, 8.1, 10])
    assert np.allclose(actual_values, expected_values)


def test_value_iteration_with_fail_prob():
    """Test value iteration on abstract Sapientino, but with small failure probability."""
    env = _RewardWrapper(5, failure_probability=0.05)
    v, policy = value_iteration(env, discount=0.9, max_iterations=100)
    actual_values = np.array(
        list(map(itemgetter(1), sorted(v.items(), key=lambda x: x[0])))
    )
    expected_values = [
        5.86562511,
        5.52781383,
        5.52781383,
        5.52781383,
        5.52781383,
        6.5516724,
    ]
    assert np.allclose(actual_values, expected_values)


def test_value_iteration_with_rb():
    """Test value iteration with the restraining bolt."""
    nb_colors = 3
    env = AbstractSapientinoTemporalGoal(nb_colors=nb_colors, failure_probability=0.1)
    v, policy = value_iteration(env, discount=0.9, max_iterations=2000)
    actual_values = np.array(
        list(map(itemgetter(1), sorted(v.items(), key=lambda x: x[0])))
    )
    expected_values = np.array(
        [
            9.33195100e00,
            1.19092409e01,
            7.11071670e-14,
            1.51983244e01,
            1.81735619e01,
            1.04840437e01,
            1.17783701e01,
            7.49535352e-14,
            1.50313099e01,
            1.91826436e01,
            9.22940208e00,
            1.33795176e01,
            7.49535352e-14,
            1.50313099e01,
            1.91826436e01,
            9.22940208e00,
            1.17783701e01,
            7.49535352e-14,
            1.70746608e01,
            1.91826436e01,
        ]
    )
    assert np.allclose(actual_values, expected_values, atol=1e-7)


def test_fluent_extraction():
    """Test fluents for this environment."""
    nb_colors = 3
    fluents = Fluents(nb_colors)

    # Colors
    assert fluents.fluents == {"red", "green", "blue"}

    # action 0 is vist
    assert fluents.evaluate(0, 1).true_propositions == set()
    assert fluents.evaluate(1, 1).true_propositions == {"red"}
    assert fluents.evaluate(2, 1).true_propositions == {"green"}
    assert fluents.evaluate(3, 1).true_propositions == {"blue"}
