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
"""Helpers related to Reward shaping wrappers."""
import gym

from multinav.helpers.gym import RewardShaper


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for reward shaping."""

    def __init__(self, env, reward_shaper: RewardShaper):
        """Initialize the Gym wrapper."""
        super().__init__(env)
        self.reward_shaper = reward_shaper

    def step(self, action):
        """Do the step."""
        state, reward, done, info = super().step(action)
        shaping_reward = self.reward_shaper.step(state, done)
        return state, reward + shaping_reward, done, info

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset(**kwargs)
        self.reward_shaper.reset(result)
        return result
