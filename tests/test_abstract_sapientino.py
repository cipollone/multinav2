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

import numpy as np
import pytest

from multinav.algorithms.value_iteration import value_iteration
from multinav.envs.abstract_sapientino import AbstractSapientino
from multinav.helpers.gym import Transition, Transitions


def test_abstract_sapientino():
    """Test abstract Sapientino."""
    env = AbstractSapientino(5)
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


def test_value_iteration():
    """Test value iteration on abstract Sapientino."""

    class _reward_wrapper(AbstractSapientino):
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

    env = _reward_wrapper(5)
    v = value_iteration(env, discount=0.9, max_iterations=200)
    assert np.allclose(v, np.array([9, 8.1, 8.1, 8.1, 8.1, 10.0]))
