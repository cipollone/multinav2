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

"""Test for callbacks."""

import gym

from multinav.wrappers.utils import CallbackWrapper


class CallbackA:
    """A generic callback."""

    def _on_step(self, action, observation, reward, done, info):
        """Any op."""
        print("observation", observation)

    def _on_reset(self, observation):
        """Any op."""
        self._last_reset = observation


def test_callback():
    """Test callbacks."""
    # Test wrapper
    env = gym.make("Taxi-v3")
    env = CallbackWrapper(env, CallbackA())

    obs = env.reset()
    assert obs == env._callback._last_reset
    env.step(0)
    env.step(0)
    env.step(0)
    obs = env.reset()
    assert obs == env._callback._last_reset
