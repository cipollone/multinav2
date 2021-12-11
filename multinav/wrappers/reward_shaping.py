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

from multinav.helpers.callbacks import Callback
from multinav.helpers.reward_shaping import RewardShaper
from multinav.wrappers.utils import CallbackWrapper


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for reward shaping.

    It sums the reward computed by a reward shaper. `self.last_experience`
    stores the last tuple returned by the original env.step.
    """

    def __init__(self, env, reward_shaper: RewardShaper):
        """Initialize the Gym wrapper."""
        super().__init__(env)
        self.reward_shaper = reward_shaper
        self.last_experience = None
        self.last_action = None

    def step(self, action):
        """Do the step."""
        state, reward, done, info = super().step(action)
        self.last_experience = state, reward, done, info
        self.last_action = action
        shaping_reward = self.reward_shaper.step(state, reward, done)
        return state, reward + shaping_reward, done, info

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset(**kwargs)
        self.reward_shaper.reset(result)
        self.last_experience = None
        self.last_action = None
        return result


class UnshapedEnv(gym.Env):
    """An empty environment without reward shaping.

    This is an empty environment. It doesn't define any dynamics, it only
    provides a view on a `RewardShapingWrapper`. At each step, the agent must
    pass the same action that was previously passed to the env wrapped by
    `RewardShapingWrapper`. The values returned are always the same at that
    environment, but the reward has no reward shaping applied.
    """

    def __init__(self, shaped_env: RewardShapingWrapper):
        """Initialize.

        :param shaped_env: any env wrapped by a RewardShapingWrapper
        """
        if not isinstance(shaped_env, RewardShapingWrapper):
            raise TypeError("Expecting a RewardShapingWrapper")
        self.shaped_env = shaped_env

        # Aliases
        self.metadata = shaped_env.metadata
        self.reward_range = shaped_env.reward_range
        self.action_space = shaped_env.action_space
        self.observation_space = shaped_env.observation_space
        self.render = shaped_env.render
        self.close = shaped_env.close
        self.seed = shaped_env.seed

    def step(self, action):
        """gym.Env step."""
        assert (
            action == self.shaped_env.last_action
        ), "Expecting the same action as the original env"
        return self.shaped_env.last_experience

    def reset(self):
        """No op."""
        assert self.shaped_env.last_experience is None


class UnshapedEnvWrapper(CallbackWrapper):
    """Make an UnshapedEnv move along with another environment.

    Wrap this to an environment that you want to use. This moves the
    unshaped_env just as the shaped_env.
    """

    def __init__(
        self,
        shaped_env: RewardShapingWrapper,
        unshaped_env: UnshapedEnv,
    ):
        """Initialize.

        :param shaped_env: the main environment to wrap (a reward shaped env).
        :param unshaped_env: a UnshapedEnv that is moved passively.
        """
        self.unshaped_env = unshaped_env
        CallbackWrapper.__init__(
            self,
            env=shaped_env,
            callback=self.MoveEnvCallback(self.unshaped_env),
        )

    def get_reward(self) -> float:
        """Return the last reward."""
        assert (
            self.unshaped_env.shaped_env.last_experience is not None
        ), "No experience yet"
        return self.unshaped_env.shaped_env.last_experience[1]

    class MoveEnvCallback(Callback):
        """Callback to move an environment and discard the output."""

        def __init__(self, env: gym.Env):
            """Initialize.

            :param env: environment to move.
            """
            Callback.__init__(self)
            self._env = env

        def _on_reset(self, obs):
            """Reset."""
            self._env.reset()

        def _on_step(self, action, observation, reward, done, info):
            """Step."""
            self._env.step(action)


class RewardShift(gym.RewardWrapper):
    """Add a constant to each reward."""

    def __init__(self, env: gym.Env, const: float):
        """Initialize."""
        super().__init__(env)
        self._const = const

    def reward(self, reward: float) -> float:
        """Just add a constant."""
        return reward + self._const
