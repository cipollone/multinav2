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
"""Helpers related to TempRL wrappers."""

from typing import List

import numpy as np
from gym import Env, ObservationWrapper
from gym.spaces import Box, Discrete, MultiDiscrete
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper

from multinav.helpers.gym import combine_boxes
from multinav.helpers.notebooks import automaton_to_rgb


class MyTemporalGoalWrapper(TemporalGoalWrapper):
    """
    Custom version of TemporalGoalWrapper.

    In particular:
    - it changes the rendering, by concatenating the frame of the environment
      and the frame of the automata.
    - if the agent goes to an automaton accepting state, the training finishes.
    """

    def __init__(
        self,
        env: Env,
        temp_goals: List[TemporalGoal],
        end_on_success: bool = True,
        end_on_failure: bool = False,
    ):
        """Initialize.

        :param env: gym environment to wrap.
        :param temp_goals: list of temporal goals.
        :param end_on_success: if true, episode terminates when the agent
            reaches the reward.
        :param end_on_failure: if true, episode terminates when the agent
            reaches a failure state.
        """
        # Super
        TemporalGoalWrapper.__init__(self, env=env, temp_goals=temp_goals)

        # Store
        self.__end_on_success = end_on_success
        self.__end_on_failure = end_on_failure

    # def render(self, mode="human", **kwargs):
    #     """
    #     Render a temporal goal environment.

    #     It consists in stacking horizontally the
    #     environment frames and all the automata frames.
    #     """
    #     if mode == "human":
    #         super().render(mode)
    #         return
    #     assert mode == "rgb_array", "Only rgb_array mode is supported."
    #     env_frame = super().render(mode, **kwargs)
    #     automata_frames = [
    #         automaton_to_rgb(
    #             tg.automaton, states2colors={tg._simulator._cur_state: "lightsalmon"}
    #         )
    #         for tg in self.temp_goals
    #     ]
    #     frames = [env_frame] + automata_frames
    #     frames = list(_add_channel(frame) for frame in frames)
    #     max_height = max(map(lambda arr: arr.shape[0], frames))
    #     # pad all frames with 4 channels of zeros
    #     for i in range(len(frames)):
    #         height, width, nb_channels = frames[i].shape
    #         pad_height = max_height - height
    #         padding = np.zeros((pad_height, width, nb_channels), dtype=np.uint8)
    #         padding.fill(255)
    #         frames[i] = np.append(frames[i], padding, axis=0)

    #     result = frames[0]
    #     for i in range(1, len(frames)):
    #         result = np.append(result, frames[i], axis=1)

    #     return result

    def step(self, action):
        """Do the step."""
        # Step
        state, reward, done, info = super().step(action)
        for tg in self.temp_goals:
            if tg.is_true():
                reward += tg.reward

        # Termination
        failure_done = self.__end_on_failure and all(
            tg.is_failed() for tg in self.temp_goals
        )
        success_done = self.__end_on_success and all(
            tg.is_true() for tg in self.temp_goals
        )
        done = done or failure_done or success_done

        return state, reward, done, info


class FlattenAutomataStates(ObservationWrapper):
    """Flatten the observation space to one array.

    A state (x, [q1, q2]) becomes (x, q1, q2).
    Discrete features x.
    """

    def __init__(self, env: Env):
        """Initialize.

        :param env: gym environment to wrap.
        """
        ObservationWrapper.__init__(self, env)

        space = env.observation_space
        assert len(space) == 2, "Expected: environment state, automata states"
        assert type(space[0]) == Discrete, "Env state must be discrete"
        assert type(space[1]) == MultiDiscrete, "Automata states are discrete"

        self.observation_space = MultiDiscrete(
            (np.insert(space[1].nvec, 0, space[0].n))
        )

    def observation(self, observation):
        """Flatten."""
        env_state = observation[0]
        automata_states = tuple(observation[1])
        return (env_state,) + automata_states


class BoxAutomataStates(ObservationWrapper):
    """Flatten the observation space to one array.

    A state (x, [q1, q2]) becomes (x, q1, q2).
    Continuous features x.
    """

    def __init__(self, env: Env):
        """Initialize.

        :param env: gym environment to wrap.
        """
        ObservationWrapper.__init__(self, env)

        space = env.observation_space
        assert len(space) == 2, "Expected: environment state, automata states"
        assert type(space[0]) == Box, "Env state must be a box"
        assert type(space[1]) == MultiDiscrete, "Automata states are discrete"

        automata_highs = space[1].nvec.astype(np.float32)
        automata_lows = np.zeros_like(automata_highs)
        automata_box = Box(automata_lows, automata_highs)

        self.observation_space = combine_boxes(space[0], automata_box)

    def observation(self, observation):
        """Flatten."""
        env_state, automata_states = observation
        obs = np.concatenate((env_state, automata_states))
        return obs


def _add_channel(frame: np.ndarray):
    """
    Add channel to a frame.

    It might be needed because of different behaviours of the rendering systems.
    """
    if frame.shape[2] != 3:
        return frame
    layer = np.zeros(frame.shape[:-1] + (1,), dtype=np.uint8)
    layer.fill(255)
    return np.append(frame, layer, axis=2)
