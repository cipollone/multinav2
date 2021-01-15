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
"""Test the wrappers."""

import gym
import pytest
from stable_baselines.common.vec_env import DummyVecEnv

from multinav.wrappers.training import UnVenv


def test_unvenv():
    """Test UnVenv."""
    def make_env():
        return gym.make("CubeCrash-v0")

    venv = DummyVecEnv([make_env, make_env])
    assert venv.num_envs == 2
    with pytest.raises(ValueError):
        UnVenv(venv)

    env1 = make_env()
    venv = DummyVecEnv([lambda: env1])
    env2 = UnVenv(venv)

    env2.reset()
    env2.step(0)

    assert env2.observation_space == env1.observation_space
    assert env2.action_space == env1.action_space
