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
"""Wrappers related to the training library: stable_baselines."""

from typing import Optional, Sequence

import gym
from gym.spaces import Tuple as GymTuple
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv, VecNormalize


class NormalizeEnvWrapper(gym.Wrapper):
    """Wrap an env and normalize features and rewards.

    Using a running average, this wrapper normalizes features, and optionally
    reward, of the input environment. When the input environment
    generates tuples as observations, you can specify which of these entries
    should be normalized.
    """

    def __init__(self, env: gym.Env, entry: Optional[int] = None):
        """Initialize.

        :param env: gym environment to wrap.
        :param entry: entry to normalize
        """
        # Super
        gym.Wrapper.__init__(self, env)

        # Check input space
        self._entry = entry
        if entry is not None and type(env.observation_space) != GymTuple:
            raise ValueError("'entry' is only supported for tuple observation spaces")

        # Stable baselines normalization class
        self._venv = DummyVecEnv([lambda: env])
        self._venv = VecNormalize(
            self._venv,
            # TODO: other args
        )

        # TODO: how to unwrap obs?

    def observation(self, observation: Sequence) -> Sequence:
        """Normalize the observation.

        :param observation: the input observation
        :return: observation with normalized entry
        """
        # TODO
        pass

    def reward(self, reward: float) -> float:
        """Normalize the reward.

        :param reward: original reward
        :return: normalized reward
        """
        # TODO
        pass


class UnVenv(gym.Env):
    """Extract a venv to a normal gym environment.

    stable_baselines' venvs represent a group of parallel environments.
    When a venv only contain one environment, we can extract it and cast it
    as a normal gym.Env. This is a redefinition, so it is more properly
    an Env not a Wrapper
    """

    # Fixed properties

    def __init__(self, venv: VecEnv):
        """Initialize.

        :param venv: a vectorized environment consisting of just one env.
        """
        # Check
        if venv.num_envs != 1:
            raise ValueError("VecEnv must contain exactly one Env to unwrap.")

        # Properties
        self.inner_venv = venv
        self.metadata = {"render.modes": VecEnv.metadata["render.modes"]}
        self.reward_range = (-float('inf'), float('inf'))  # least constraint
        self.action_space = venv.action_space
        self.observation_space = venv.observation_space

    def step(self, action):
        """Do a step."""
        obs, reward, done, info = self.inner_venv.step([action])
        return obs[0], reward[0], done[0], info[0]

    def reset(self):
        """Reset."""
        return self.inner_venv.reset()[0]
