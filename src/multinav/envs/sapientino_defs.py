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

"""Definitions that are shared by all sapientino-like environments.

This module does not define any environment. It contains definitions that are
shared by all sapientino-like environments.
"""

from gym_sapientino.core.types import color2int as enum_color2int

# NOTE: the map and the sequence of color must remain constant between
#   experiments. We can either define them constant like so, or receive them
#   as arguments from training (I mean, from the previous run).


# This is the sequence of colors (positions) that the agent must visit
sapientino_color_sequence = ["red", "green", "blue"]

# This is the shape and configuration of the sapientino map
sapientino_map_str = """\
|B p#######|
|## #######|
|  b       |
|## ###r###|
|##y### ###|
|## ###g###|"""
sapientino_n_rooms = 3


# These mappings are often useful (from colors to ID and vice versa)
color2int = {c.value: i for c, i in enum_color2int.items()}
int2color = {i: c for c, i in color2int.items()}
