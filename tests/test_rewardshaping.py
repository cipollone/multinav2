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

from multinav.helpers.gym import RewardShaper


def test_reward_shaper():
    """Test RewardShaper."""
    gamma = 0.5

    shaper = RewardShaper(
        value_function=lambda x: x,
        mapping_function=lambda x: x + 1,
        gamma=gamma,
        zero_terminal_state=True,
    )

    shaper.reset(-1)
    assert shaper.step(-1, False) == 0
    assert shaper.step(0, False) == 0.5
    assert shaper.step(2, False) == 0.5
    assert shaper.step(2, False) == -1.5
    assert shaper.step(2, True) == -3


def test_null_shaper():
    """Test null reward shaping."""
    shaper = RewardShaper(
        value_function=lambda _: 0.0,
        mapping_function=lambda x: x,
        gamma=0.9,
        zero_terminal_state=False,
    )

    shaper.reset(-1)
    assert shaper.step(-1, False) == 0.0
    assert shaper.step(2, False) == 0.0
    assert shaper.step(3, False) == 0.0
