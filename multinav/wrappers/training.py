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

from typing import Optional

import gym
import numpy as np
from stable_baselines.common.running_mean_std import RunningMeanStd
from stable_baselines.common.vec_env import VecEnv

from multinav.helpers.gym import Observation


class NormalizeEnvWrapper(gym.Wrapper):
    """Wrap an env and normalize features and rewards.

    Using a running average, this wrapper normalizes features of the input
    environment. When the input environment generates a tuple of observations,
    you can specify which of these entries should be normalized.
    """

    def __init__(
        self,
        env: gym.Env,
        training: bool,
        entry: Optional[int] = None,
    ):
        """Initialize.

        :param env: gym environment to wrap.
        :param training: set this to true, if averages should be updated
            (during training in fact).
        :param entry: entry to normalize (can be None if observation is a
            single array).
        """
        # Check input space
        if type(env.observation_space) not in (gym.spaces.Box, gym.spaces.Tuple):
            raise TypeError("Environment observation_space not supported")
        if entry is not None and type(env.observation_space) != gym.spaces.Tuple:
            raise ValueError("'entry' is only supported for tuple observation spaces")

        # Wrap
        gym.Wrapper.__init__(self, env)

        # Store
        self._training = training
        self._entry = entry
        self._calc_eps = 1e-8
        self._entry_shape = (
            self.observation_space.shape
            if self._entry is None
            else self.observation_space[entry].shape
        )

        # Observation to normalize
        self._observation_rms = RunningMeanStd(shape=self._entry_shape)

    def __getstate__(self) -> dict:
        """Get pickleable state."""
        state = self.__dict__.copy()
        del state["env"]  # Env is not pickleable
        return state

    def __setstate__(self, state: dict):
        """Restore the pickled state.

        :param state: a namespace.
        """
        self.__dict__.update(state)
        assert "env" not in state
        self.env = None  # To be set with set_env

    def set_env(self, env: gym.Env):
        """Set the wrapped gym env.

        This must be done after unpickling.
        :param env: the reconstructed gym environment to re-wrap.
        """
        if self.env is not None:
            raise TypeError("Environment already set")
        gym.Wrapper.__init__(self, env)

    def _standardize_vec(self, data: np.ndarray, rms: RunningMeanStd) -> np.ndarray:
        """Normalize and return an observation entry.

        :param data: data to normalize
        :param rms: object that contains the moments
        :return: standardized data
        """
        shifted_data = data - rms.mean
        standardized_data = shifted_data / np.sqrt(rms.var + self._calc_eps)
        return standardized_data

    def set_training(self, value: bool):
        """Assign a value to training (see __init__ for doc)."""
        self._training = value

    def reset(self):
        """Reset the environment."""
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action):
        """Make one action and observe."""
        observation, reward, done, info = self.env.step(action)
        observation = self.observation(observation)
        return observation, reward, done, info

    def observation(self, observation: Observation) -> Observation:
        """Normalize the observation.

        :param observation: the input observation
        :return: observation with normalized entry
        """
        # Get entry
        obs = observation if self._entry is None else observation[self._entry]

        # Normalize
        if self._training:
            obs_batch = np.expand_dims(obs, 0)
            self._observation_rms.update(obs_batch)
        obs = self._standardize_vec(obs, self._observation_rms)

        # Set entry
        if self._entry is None:
            observation = obs
        else:
            observation = list(observation)
            observation[self._entry] = obs
            observation = tuple(observation)

        return observation


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
        self.action_space = venv.action_space
        self.observation_space = venv.observation_space

    def step(self, action):
        """Do a step."""
        obs, reward, done, info = self.inner_venv.step([action])
        return obs[0], reward[0], done[0], info[0]

    def reset(self):
        """Reset."""
        return self.inner_venv.reset()[0]
