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

from typing import Any, Dict, Optional

from typing_extensions import Protocol

from multinav.helpers.misc import Saver

# Types
Observation = Any
Action = int


class Callback(Protocol):
    """Abstract interface of callbacks.

    To implement a callback, you can either subclass this or just
    declare an indipendent class with the methods declared below.
    """

    def _on_reset(self, obs: Observation) -> None:
        """Do this on reset.

        :param obs: observation received from env.reset
        """
        raise NotImplementedError("Abstract")

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


class SaverCallback(Saver, Callback):
    """This is a callback that periodically saves."""

    def __init__(self, *, save_freq: Optional[int], **saver_kwargs):
        """Initialize.

        :param save_freq: number of steps between each save (None means never).
        """
        Saver.__init__(self, **saver_kwargs)
        self._save_freq = save_freq
        self.num_timesteps = 0

    def _on_reset(self, obs: Observation) -> None:
        """Nothing to do."""
        pass

    def _on_step(
        self,
        action: Action,
        obs: Observation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """See Callback._on_step."""
        self.num_timesteps += 1
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save()

    def save(self):
        """Save a checkpoint now."""
        Saver.save(self, step=self.num_timesteps)
