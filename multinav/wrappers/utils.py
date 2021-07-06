# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#
"""Utilities for the OpenAI Gym wrappers."""
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional

import gym
import matplotlib.pyplot as plt
from gym import Wrapper
from gym.spaces import Tuple as GymTuple
from PIL import Image

from multinav.helpers.callbacks import Callback

logger = logging.getLogger(__name__)


class MyMonitor(Wrapper):
    """A simple monitor."""

    def __init__(self, env: gym.Env, directory: str, force: bool = False):
        """
        Initialize the environment.

        :param env: the environment.
        :param directory: the directory where to save elements.
        """
        super().__init__(env)

        self._directory = Path(directory)
        shutil.rmtree(directory, ignore_errors=force)
        self._directory.mkdir(exist_ok=False)

        self._current_step = 0
        self._current_episode = 0

    def _save_image(self):
        """Save a frame."""
        array = self.render(mode="rgb_array")
        image = Image.fromarray(array)
        episode_dir = f"{self._current_episode:05d}"
        filepath = f"{self._current_step:05d}.jpeg"
        (self._directory / episode_dir).mkdir(parents=True, exist_ok=True)
        image.save(str(self._directory / episode_dir / filepath))

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset(**kwargs)
        self._current_step = 0
        self._current_episode += 1
        self._save_image()
        return result

    def step(self, action):
        """Do a step in the environment, and record the frame."""
        result = super().step(action)
        self._current_step += 1
        self._save_image()
        return result


class MyStatsRecorder(gym.Wrapper):
    """Stats recorder."""

    def __init__(self, env: gym.Env, gamma: float, prefix: str = ""):
        """
        Initialize stats recorder.

        :param env: the environment to monitor.
        :param prefix: the prefix to add to statistics attributes.
        """
        super().__init__(env)
        self._prefix = prefix
        self._gamma = gamma
        self._episode_lengths: List[int] = []
        self._episode_rewards: List[float] = []
        self._episode_returns: List[float] = []
        self._timestamps: List[float] = []
        self._steps = None
        self._total_steps = 0
        self._rewards = None
        self._discount = None
        self._returns = None
        self._done = False

        # Extra attributes
        self._episode_td_max: List[float] = []
        self._td_max = 0.0

        self._set_attributes()

    def _set_attributes(self):
        """Set main attributes with the prefix."""
        setattr(self, self._prefix + "episode_lengths", self._episode_lengths)
        setattr(self, self._prefix + "episode_rewards", self._episode_rewards)
        setattr(self, self._prefix + "episode_returns", self._episode_returns)
        setattr(self, self._prefix + "total_steps", self._total_steps)
        setattr(self, self._prefix + "timestamps", self._timestamps)

        setattr(self, self._prefix + "episode_td_max", self._episode_td_max)

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        self._steps += 1
        self._total_steps += 1
        self._rewards += reward
        self._returns += self._discount * reward
        self._discount *= self._gamma
        self._done = done
        if done:
            self.save_complete()

        logger.debug(f"state {state}, reward {reward}, done {done}, info {info}")

        return state, reward, done, info

    def save_complete(self):
        """Save episode statistics."""
        if self._steps is not None:
            self._episode_lengths.append(self._steps)
            self._episode_rewards.append(float(self._rewards))
            self._episode_returns.append(float(self._returns))
            self._timestamps.append(time.time())

            self._episode_td_max.append(self._td_max)

    def reset(self, **kwargs):
        """Do reset."""
        result = super().reset(**kwargs)
        self._done = False
        self._steps = 0
        self._rewards = 0
        self._discount = 1.0
        self._returns = 0.0

        self._td_max = 0.0

        logger.debug(f"reset state {result}")

        return result

    def update_extras(self, *, td: Optional[float] = None):
        """Update the value of extra logs.

        These variables are upted with the given values.
        At the end of the episode they will be available along with other
        logs. This is necessary for quantities that cannot be computed
        from the environment.

        :param td: temporal difference error
        """
        if td is not None and td > self._td_max:
            self._td_max = td


class SingleAgentWrapper(Wrapper):
    """
    Wrapper for multi-agent OpenAI Gym environment to make it single-agent.

    It adapts a multi-agent OpenAI Gym environment with just one agent
    to be used as a single agent environment.
    In particular, this means that if the observation space and the
    action space are tuples of one space, the new
    spaces will remove the tuples and return the unique space.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the wrapper."""
        super().__init__(*args, **kwargs)

        self.observation_space = self._transform_tuple_space(self.observation_space)
        self.action_space = self._transform_tuple_space(self.action_space)

    def _transform_tuple_space(self, space: GymTuple):
        """Transform a Tuple space with one element into that element."""
        assert isinstance(
            space, GymTuple
        ), "The space is not an instance of gym.spaces.tuples.Tuple."
        assert len(space.spaces) == 1, "The tuple space has more than one subspaces."
        return space.spaces[0]

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step([action])
        new_state = state[0]
        return new_state, reward, done, info

    def reset(self, **kwargs):
        """Do a step."""
        state = super().reset(**kwargs)
        new_state = state[0]
        return new_state


class CallbackWrapper(Wrapper):
    """Inject callbacks in the training algorithm."""

    def __init__(
        self,
        env: gym.Env,
        callback: Callback,
    ):
        """Initialize.

        :param env: Gym environment to wrap
        :param callback: a callback object
        """
        Wrapper.__init__(self, env)
        self._callback = callback

    def reset(self):
        """Reset the environment."""
        obs = Wrapper.reset(self)
        self._callback._on_reset(obs)
        return obs

    def step(self, action):
        """Environment step."""
        observation, reward, done, info = Wrapper.step(self, action)
        self._callback._on_step(action, observation, reward, done, info)
        return observation, reward, done, info


class AbstractSapientinoRenderer(Wrapper):
    """Wraps and display abstract sapientino."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        Wrapper.__init__(self, *args, **kwargs)
        self.__img = None

    def render(self, mode="human"):
        """Show on screen the sapientino images.

        mode is ignored.
        """
        # Image array
        img_array = self.env.render(mode="rgb_array")
        # Draw
        if self.__img is None:
            self.__img = plt.imshow(img_array)
        else:
            self.__img.set_data(img_array)
        plt.pause(0.01)
        plt.draw()


class Renderer(Wrapper):
    """Just render the env while executing."""

    def reset(self, **kwargs):
        """Reset the env."""
        obs = self.env.reset(**kwargs)
        self.env.render()
        return obs

    def step(self, action):
        """Gym interfact for step."""
        ret = self.env.step(action)
        self.env.render()
        return ret
