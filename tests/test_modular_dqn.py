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
import tempfile
from pathlib import Path

import gym
import numpy as np
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy

from multinav.algorithms.modular_dqn import ModularPolicy
from multinav.envs.grid_sapientino import generate_grid
from multinav.helpers.gym import rollout
from multinav.restraining_bolts.rb_grid_sapientino import GridSapientinoRB
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import MyStatsRecorder, SingleAgentWrapper

# def test_dqn():
#     """Test the ModularDQN agent."""
#     MAX_EPISODE_STEPS = 200.0
#     env = gym.make("MountainCar-v0")
#     model = DQN(MlpPolicy, env, verbose=1)
#     model.learn(total_timesteps=100000)
#
#     env = MyStatsRecorder(env)
#     rollout(env, nb_episodes=10, policy=lambda env, state: model.predict(state)[0])
#     env.close()
#
#     # assert that we reach the goal sometimes
#     logging.debug(f"episode rewards: {env.episode_rewards}")
#     logging.debug(f"episode lengths: {env.episode_lengths}")
#     assert np.mean(env.episode_lengths) != MAX_EPISODE_STEPS
#
#
# def test_sapientino_dqn():
#     """Test DQN on Sapientino agent."""
#     nb_colors = 2
#     temp_file = Path(tempfile.mktemp(suffix=".txt"))
#     generate_grid(nb_colors, temp_file)
#     agent_configuration = SapientinoAgentConfiguration(continuous=True)
#     configuration = SapientinoConfiguration(
#         [agent_configuration],
#         path_to_map=temp_file,
#         reward_per_step=-0.01,
#         reward_outside_grid=0.0,
#         reward_duplicate_beep=0.0,
#     )
#     env = SingleAgentWrapper(SapientinoDictSpace(configuration))
#     tg = GridSapientinoRB(nb_colors).make_sapientino_goal()
#     env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
#
#     model = DQN(MlpPolicy, env, verbose=1)
#     model.learn(total_timesteps=1000)
#     env = MyStatsRecorder(env)
#     rollout(
#         env,
#         nb_episodes=1,
#         policy=lambda env, state: model.predict(state)[0],
#         callback=lambda env, *args: env.render(),
#     )
#     env.close()
#
#     # assert that we reach the goal sometimes
#     logging.debug(f"episode rewards: {env.episode_rewards}")
#     logging.debug(f"episode lengths: {env.episode_lengths}")
#
#
# def test_modular_dqn():
#     """Test the ModularDQN agent."""
#     nb_colors = 2
#     temp_file = Path(tempfile.mktemp(suffix=".txt"))
#     generate_grid(nb_colors, temp_file)
#     agent_configuration = SapientinoAgentConfiguration(continuous=True)
#     configuration = SapientinoConfiguration(
#         [agent_configuration],
#         path_to_map=temp_file,
#         reward_per_step=-0.01,
#         reward_outside_grid=0.0,
#         reward_duplicate_beep=0.0,
#     )
#     env = SingleAgentWrapper(SapientinoDictSpace(configuration))
#     tg = GridSapientinoRB(nb_colors).make_sapientino_goal()
#     env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
#
#     model = DQN(ModularPolicy, env, verbose=1)
#     model.learn(total_timesteps=100000)
#
#     env = MyStatsRecorder(env)
#     rollout(env, nb_episodes=10, policy=lambda env, state: model.predict(state)[0])
#     env.close()
#
#     # assert that we reach the goal sometimes
#     logging.debug(f"episode rewards: {env.episode_rewards}")
#     logging.debug(f"episode lengths: {env.episode_lengths}")
