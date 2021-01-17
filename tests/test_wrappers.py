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

import pickle
import tempfile

import gym
import numpy as np
import pytest
from stable_baselines.common.vec_env import DummyVecEnv

from multinav.wrappers.training import NormalizeEnvWrapper, UnVenv


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


class DuplicateObsWrapper(gym.ObservationWrapper):
    """Class to duplicate the observations."""

    def __init__(self, env):
        """Initialize."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Tuple(
            (env.observation_space, env.observation_space)
        )

    def observation(self, observation):
        """Process observation."""
        return (observation, observation)


def test_normalizer():
    """Test NormalizeEnvWrapper."""
    env1 = gym.make("BreakoutDeterministic-v4")

    # Not a tuple environment
    with pytest.raises(ValueError):
        NormalizeEnvWrapper(env1, True, 0)

    env2 = NormalizeEnvWrapper(env1, True)

    # Average updated
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert np.any(obs2 != obs1)
    assert np.allclose(env2._observation_rms.mean, obs1, rtol=0.001)

    # Save and load
    with tempfile.TemporaryFile() as fp:
        pickle.dump(env2, fp)
        fp.seek(0)
        env3 = pickle.load(fp)

    assert env3.env is None
    env3._training = False
    env3.set_env(env1)
    env3.step(0)
    assert np.all(env3._observation_rms.mean == env2._observation_rms.mean)
    assert np.all(env3._observation_rms.var == env2._observation_rms.var)

    # Normalize just one entry
    env4 = DuplicateObsWrapper(env1)
    env5 = NormalizeEnvWrapper(env4, training=True, entry=1)
    assert env5.observation_space == env4.observation_space

    obs3 = env5.reset()
    assert np.all(env5._observation_rms.mean == env3._observation_rms.mean)
    assert np.all(obs3[0] == obs1)
    assert np.all(obs3[1] == obs2)

    # Freeze weights
    env5.set_training(False)
    env5.step(1)
    env5.step(0)
    obs4, _, _, _ = env5.step(0)
    assert np.any(obs4[1] != obs3)
    assert np.all(env5._observation_rms.mean == env3._observation_rms.mean)
