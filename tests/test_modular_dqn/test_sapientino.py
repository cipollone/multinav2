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
"""Tests for modular DQN policy with temporal goals."""
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from flloat.semantics import PLInterpretation
from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from stable_baselines import DQN

from multinav.algorithms.modular_dqn import ModularPolicy
from multinav.helpers.gym import rollout
from multinav.restraining_bolts.automata import make_sapientino_goal_with_automata
from multinav.wrappers.sapientino import ContinuousRobotFeatures
from multinav.wrappers.temprl import MyTemporalGoalWrapper
from multinav.wrappers.utils import MyStatsRecorder, SingleAgentWrapper


class TestSapientino4ColorsSquare:
    """Test Sapientino in a square with 4 colors."""

    SQUARE_MAP = """\
        |
     gb |
     ry |
        |"""

    COLORS = ["red", "green"]  # "blue", "yellow"]

    @pytest.fixture(autouse=True)
    def _disable_debug_logging(self, disable_debug_logging):
        """Disable debug logging."""

    @classmethod
    def extract_sapientino_fluents(cls, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        is_beep = obs.get("beep") > 0
        color_id = obs.get("color")
        if is_beep and 0 <= color_id - 1 < len(cls.COLORS):
            color = cls.COLORS[color_id - 1]
            fluents = {color} if color in cls.COLORS else set()
        else:
            fluents = set()
        return PLInterpretation(fluents)

    def test_run(self):
        """Test sapientino with 4 colors in a square."""
        agent_configuration = SapientinoAgentConfiguration(
            continuous=False, initial_position=(1, 2)
        )
        temp_file = Path(tempfile.mktemp(suffix=".txt"))
        temp_file.write_text(self.SQUARE_MAP)
        configuration = SapientinoConfiguration(
            [agent_configuration],
            path_to_map=temp_file,
            reward_per_step=-0.01,
            reward_outside_grid=-1.0,
            reward_duplicate_beep=0.0,
        )
        env = SingleAgentWrapper(SapientinoDictSpace(configuration))
        tg = make_sapientino_goal_with_automata(
            self.COLORS, self.extract_sapientino_fluents, reward=10.0
        )
        env = ContinuousRobotFeatures(MyTemporalGoalWrapper(env, [tg]))
        print(f"Observation space: {env.observation_space}")
        learn_env = MyStatsRecorder(TimeLimit(env, max_episode_steps=30))

        model = DQN(
            ModularPolicy,
            learn_env,
            verbose=1,
            learning_starts=5000,
            exploration_final_eps=0.1,
            policy_kwargs=dict(layers=[64, 64]),
        )

        model.learn(total_timesteps=25000)

        logging.debug("done!")
        logging.debug(f"episode rewards: {learn_env.episode_rewards}")
        logging.debug(f"episode lengths: {learn_env.episode_lengths}")

        test_env = MyStatsRecorder(TimeLimit(env, max_episode_steps=30))
        rollout(
            test_env, nb_episodes=3, policy=lambda env, state: model.predict(state)[0]
        )
        test_env.close()

        logging.debug("done!")
        logging.debug(f"episode rewards: {test_env.episode_rewards}")
        logging.debug(f"episode lengths: {test_env.episode_lengths}")
        assert np.isclose(np.mean(test_env.episode_rewards), 9.9)
