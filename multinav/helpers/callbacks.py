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

from typing import Any, Callable, Dict, List, Optional

from typing_extensions import Protocol

from multinav.helpers.gym import Action, Done, Reward, State
from multinav.helpers.misc import Saver

StepFunction = Callable[[Action, State, Reward, Done, Dict[str, Any]], None]
EpisodeFunction = Callable[[State, int], None]


class Callback(Protocol):
    """Abstract interface of callbacks.

    To implement a callback, you can either subclass this or just
    declare an indipendent class with the methods declared below.
    """

    def _on_reset(self, obs: State) -> None:
        """Do this on reset.

        :param obs: observation received from env.reset
        """
        raise NotImplementedError("Abstract")

    def _on_step(
        self,
        action: Action,
        obs: State,
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

    def _on_reset(self, obs: State) -> None:
        """Nothing to do."""
        pass

    def _on_step(
        self,
        action: Action,
        obs: State,
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


class FnCallback(Callback):
    """Periodically calls a function."""

    def __init__(
        self,
        *,
        step_fn: Optional[StepFunction] = None,
        step_freq: Optional[int] = None,
        ep_fn: Optional[EpisodeFunction] = None,
        ep_freq: Optional[int] = None,
    ):
        """Initialize.

        :param step_fn: Function to call for step callback
        :param step_freq: Call frequency in number of steps.
        :param ep_fn: Function to call for episode callback.
        :param ep_freq: Call frequency in number of episodes.
        """
        assert not step_freq or step_fn, "No step function"
        assert not ep_freq or ep_fn, "No episode function"

        self._step_fn = step_fn
        self._step_freq = step_freq
        self._ep_fn = ep_fn
        self._ep_freq = ep_freq

        self.num_timesteps = 0
        self.num_episodes = 0

    def _ep_call(self, *args, **kwargs):
        """Just remove self."""
        self._ep_fn(*args, **kwargs)

    def _step_call(self, *args, **kwargs):
        """Just remove self."""
        self._step_fn(*args, **kwargs)

    def _on_reset(self, obs: State) -> None:
        """Maybe call function on reset."""
        self.num_episodes += 1
        if self._ep_freq:
            if self.num_episodes % self._ep_freq == 0:
                self._ep_call(obs, self.num_episodes)

    def _on_step(
        self,
        action: Action,
        obs: State,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Maybe call function on step."""
        self.num_timesteps += 1
        if self._step_freq:
            if self.num_timesteps % self._step_freq == 0:
                info["num_episodes"] = self.num_episodes
                self._step_call(action, obs, reward, done, info)


class CallbackList(Callback):
    """Calls a sequence of callbacks."""

    def __init__(self, callbacks: List[Callback]):
        """Initialize."""
        self._callbacks = callbacks

    def _on_reset(self, obs: State) -> None:
        """Signal reset to all."""
        for callback in self._callbacks:
            callback._on_reset(obs)

    def _on_step(
        self,
        action: Action,
        obs: State,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Do step on all."""
        for callback in self._callbacks:
            callback._on_step(action, obs, reward, done, info)
