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
"""Tests for modular DQN policy."""

import logging

import gym
import numpy as np
from gym.spaces import Box
from stable_baselines import DQN

from multinav.algorithms.modular_dqn import ModularPolicy
from multinav.helpers.gym import rollout
from multinav.wrappers.utils import MyStatsRecorder


def test_dqn():
    """Test the ModularDQN agent."""
    MAX_EPISODE_STEPS = 200.0

    class wrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            box = env.observation_space
            lows = box.low
            highs = box.high
            new_box = Box(np.append(lows, 0.0), np.append(highs, 0.0))
            self.observation_space = new_box

        def step(self, action):
            state, reward, done, info = super().step(action)
            new_state = np.append(state, [0.0])
            return new_state, reward, done, info

        def reset(self):
            state = super().reset()
            return np.append(state, [0.0])

    env = gym.make("MountainCar-v0")
    env = wrapper(env)
    model = DQN(ModularPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

    env = MyStatsRecorder(env)
    rollout(env, nb_episodes=10, policy=lambda env, state: model.predict(state)[0])
    env.close()

    # assert that we reach the goal sometimes
    logging.debug(f"episode rewards: {env.episode_rewards}")
    logging.debug(f"episode lengths: {env.episode_lengths}")
    assert np.mean(env.episode_lengths) != MAX_EPISODE_STEPS
