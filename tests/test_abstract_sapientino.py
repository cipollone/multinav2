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
from multinav.envs.abstract_sapientino import (
    AbstractSapientino,
    AbstractSapientinoTemporalGoal,
)
from multinav.helpers.gym import Transition, Transitions
from multinav.restraining_bolts.rb_abstract_sapientino import AbstractSapientinoRB


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
    rb = AbstractSapientinoRB(nb_colors)
    env = AbstractSapientinoTemporalGoal(rb, [nb_colors], dict(failure_probability=0.0))
    v, policy = value_iteration(env, discount=0.9, max_iterations=200)
    actual_values = np.array(
        list(map(itemgetter(1), sorted(v.items(), key=lambda x: x[0])))
    )
    expected_values = np.array(
        [
            4.3046720115533725,
            5.904899911553373,
            2.647284498601486e-14,
            8.099999911553372,
            9.999999911553372,
            0.0,
            4.782968911553372,
            5.314409911553372,
            2.647284498601486e-14,
            7.289999911553372,
            9.999999911553372,
            0.0,
            3.874204801553371,
            6.560999911553372,
            2.647284498601486e-14,
            7.289999911553372,
            9.999999911553372,
            0.0,
            3.874204801553371,
            5.314409911553372,
            2.647284498601486e-14,
            8.999999911553372,
            9.999999911553372,
            0.0,
        ]
    )
    assert np.allclose(actual_values, expected_values)
