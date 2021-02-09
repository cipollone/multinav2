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

"""Test algorithm implementations."""
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from gym.wrappers import TimeLimit

from multinav.algorithms.q_learning import QLearning
from multinav.helpers.gym import rollout
from multinav.wrappers.utils import MyStatsRecorder


def test_q_learning():
    """Test q-learning."""
    env = TimeLimit(FrozenLakeEnv(is_slippery=False), max_episode_steps=100)
    Q = QLearning(
        env=env,
        total_timesteps=100000,
        alpha=0.1,
        eps=0.8,
        gamma=1.0,
    ).learn()
    env = MyStatsRecorder(env, gamma=1.0)
    rollout(env, nb_episodes=10, policy=lambda _, _state: np.argmax(Q[_state]))
    assert np.mean(env.episode_rewards) == 1.0
