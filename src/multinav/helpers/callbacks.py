# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#
"""Callbacks definitions.

This module defines callbacks and their interface. Callback is a generic
concept, these are not specific to the learning library.
Callbacks can be applied with multinav.wrappers.CallbackWrapper.
"""

from abc import abstractmethod
from typing import Any, Dict

from multinav.helpers.general import ABCWithMethods

# Types
Observation = Any
Action = int


class CallbackInterface(ABCWithMethods):
    """Interface of callbacks.

    To implement a callback, you can either subclass this or just
    declare an indipendent class with the methods declared below.
    """

    # Required methods
    _abs_methods = ["_on_step", "_on_reset"]

    @abstractmethod
    def _on_reset(self, obs: Observation) -> None:
        """Do this on reset.

        :param obs: observation received from env.reset
        """
        raise NotImplementedError("Abstract")

    @abstractmethod
    def _on_step(
        self,
        action: Action,
        obs: Observation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Do this on each step.

        :param action: selected action
        :param obs: returned observation
        :param reward: returned reward
        :param done: returned done flag
        :param info: returned dict of infos
        """
        raise NotImplementedError("Abstract")
