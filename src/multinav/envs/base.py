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

"""Common definitions for environments.

This module defines types used by all environments. No environment is
defined here.
"""

from abc import abstractmethod
from typing import Any

from flloat.semantics import PLInterpretation

from multinav.helpers.general import ABC2, AbstractAttribute


class AbstractFluents(ABC2):
    """Definition of fluents for an environment.

    This is the base class for a features extractor in an environment.
    `fluents` is the set of fluents that are valuated.  The interpretation
    returned by valuate can only contain symbols from this set.
    """

    fluents = AbstractAttribute()  # type: Any

    @abstractmethod
    def valuate(self, obs, action) -> PLInterpretation:
        """Compute the current propositional interpretation.

        This function also respects the interface defined in
        temprl.wrapper.TemporalGoal.
        :param obs: environment observation
        :param action: last executed action
        :return: a valuation for all fluents
        """
        pass
