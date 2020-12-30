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
"""Restraining bolt for Abstract Sapientino."""
from flloat.semantics import PLInterpretation

from multinav.restraining_bolts.base import SapientinoRB


class AbstractSapientinoRB(SapientinoRB):
    """Restraining Bolt for abstract Sapientino."""

    def extract_sapientino_fluents(self, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        # see SapientinoRB
        observation_offset = 1
        visit_action = 1

        color_id = obs - observation_offset
        if visit_action == action:
            fluents = {self.colors[color_id]}
        else:
            fluents = set()
        return PLInterpretation(fluents)
