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

"""Tests for Grid Sapientino."""

import numpy as np
from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.types import Colors

from multinav.algorithms.q_learning import QLearning
from multinav.envs.env_cont_sapientino import Fluents
from multinav.envs.temporal_goals import SapientinoGoal
from multinav.helpers.gym import rollout
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import MyStatsRecorder, SingleAgentWrapper


def test_grid_sapientino_rb_q_learning(disable_debug_logging):
    """Test grid Sapientino + RB, using Q-Learning."""
    configuration = SapientinoConfiguration(
        reward_per_step=-0.01,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))
    tg = SapientinoGoal(
        colors=["red", "green", "blue"],
        fluents=Fluents(colors_set={str(c) for c in Colors}),
        reward=10.0,
    )
    env = GridRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
    env = TimeLimit(env, max_episode_steps=25)

    Q = QLearning(env, total_timesteps=40000).learn()

    env = MyStatsRecorder(env, gamma=0.9)
    rollout(env, policy=lambda _env, state: np.argmax(Q[state]))

    assert np.mean(env.episode_rewards) == 9.9
