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
"""Tests for gym helpers."""

import numpy as np
import pytest
from gym.spaces import Box

from multinav.helpers.gym import combine_boxes


def test_combine_boxes():
    """Test combine_boxes."""
    b1 = Box(-1.0, 1.0, shape=(3, 3))

    with pytest.raises(AssertionError):
        combine_boxes(b1)

    b1 = Box(-1.0, 1.0, shape=[3])
    bb1 = combine_boxes(b1)
    assert b1 == bb1

    b2 = Box(-2, 3, shape=[2])
    bb2 = Box(
        low=np.array([-1, -1, -1, -2, -2], dtype=np.float32),
        high=np.array([1, 1, 1, 3, 3], dtype=np.float32),
    )
    assert combine_boxes(b1, b2) == bb2

    b3 = Box(0, 5, shape=[1], dtype=np.int8)
    assert combine_boxes(b1, b2, b3) == combine_boxes(bb2, b3)
