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

"""Test fluents extraction."""

import pytest

from multinav.envs.base import AbstractFluents


class Fluents1(AbstractFluents):
    def valuate(self, obs, action):
        return {"is_one"} if obs == 1 else set()


class Fluents2(Fluents1):
    def __init__(self):
        self.fluents = {"is_one", "is_two"}


def test_base():
    """Test the base class."""
    with pytest.raises(TypeError):
        fluents = Fluents1()
    fluents = Fluents2()

    assert fluents.valuate(1, 0).issubset(fluents.fluents)
