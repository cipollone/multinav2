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

from flloat.semantics import PLInterpretation

from multinav.envs.base import AbstractFluents
from multinav.envs.cont_sapientino import Fluents as ContSapientinoFluents


class Fluents1(AbstractFluents):
    def valuate(self, obs, action):
        return {"is_one"} if obs == 1 else set()


class Fluents2(Fluents1):
    def __init__(self):
        self.fluents = {"is_one", "is_two"}


def test_fluents_base():
    """Test the base class."""
    with pytest.raises(TypeError):
        fluents = Fluents1()
    fluents = Fluents2()

    assert fluents.valuate(1, 0).issubset(fluents.fluents)


def test_fluents_cont_sapientino():
    """Test fluents extraction on cont-sapientino."""
    # NOTE: this test depends on gym-sapientino color order
    with pytest.raises(ValueError):
        fluents = ContSapientinoFluents({"not a color"})
    fluents = ContSapientinoFluents({"red", "blue"})  # with just 2 fluents

    assert fluents.valuate(dict(beep=0, color=1), 0) == PLInterpretation(set())
    assert fluents.valuate(dict(beep=1, color=1), 0) == PLInterpretation({"red"})
    assert fluents.valuate(dict(beep=1, color=3), 0) == PLInterpretation({"blue"})
    with pytest.raises(RuntimeError):
        fluents.valuate(dict(beep=1, color=2), 0)   # green not used
