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

"""Tests for temporal goals module."""

import pytest
from flloat.semantics import PLInterpretation

from multinav.envs import temporal_goals
from multinav.envs.env_cont_sapientino import Fluents as SapientinoFluents


def test_sapientino_goal():
    """Test sapientino goal."""
    fluents = SapientinoFluents({"red", "green", "blue"})
    with pytest.raises(ValueError):
        temporal_goals.SapientinoGoal(
            colors=["red", "yellow"],
            fluents=fluents,
            reward=1.0,
        )
    tg = temporal_goals.SapientinoGoal(
        colors=["red", "blue"],
        fluents=fluents,
        reward=1.0,
    )

    # some constants
    empty = PLInterpretation({})
    red = PLInterpretation({"red"})
    blue = PLInterpretation({"blue"})
    yellow = PLInterpretation({"yellow"})

    assert len(tg.automaton.states) == 4
    assert tg.automaton.accepts([empty, red, empty, blue])
    assert tg.automaton.accepts([empty, red, empty, blue, empty])
    assert not tg.automaton.accepts([red, yellow])
    assert not tg.automaton.accepts([red])
    assert not tg.automaton.accepts([])
