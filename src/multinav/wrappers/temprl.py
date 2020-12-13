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
import numpy as np
from temprl.wrapper import TemporalGoalWrapper

from multinav.helpers.notebooks import automaton_to_rgb


class MyTemporalGoalWrapper(TemporalGoalWrapper):
    """
    Custom version of TemporalGoalWrapper.

    In particular:
    - it changes the rendering, by concatenating the frame of the environment
      and the frame of the automata.
    - if the agent goes to an automaton accepting state, the training finishes.
    """

    def render(self, mode="human", **kwargs):
        """
        Render a temporal goal environment.

        It consists in stacking horizontally the
        environment frames and all the automata frames.
        """
        assert mode == "rgb_array", "Only rgb_array mode is supported."
        env_frame = super().render(mode, **kwargs)
        automata_frames = [
            automaton_to_rgb(
                tg.automaton, states2colors={tg._simulator.cur_state: "lightsalmon"}
            )
            for tg in self.temp_goals
        ]
        frames = [env_frame] + automata_frames
        frames = list(_add_channel(frame) for frame in frames)
        max_height = max(map(lambda arr: arr.shape[0], frames))
        # pad all frames with 4 channels of zeros
        for i in range(len(frames)):
            height, width, nb_channels = frames[i].shape
            pad_height = max_height - height
            padding = np.zeros((pad_height, width, nb_channels), dtype=np.uint8)
            padding.fill(255)
            frames[i] = np.append(frames[i], padding, axis=0)

        result = frames[0]
        for i in range(1, len(frames)):
            result = np.append(result, frames[i], axis=1)

        return result

    def step(self, action):
        """Do the step."""
        state, reward, done, info = super().step(action)
        for tg in self.temp_goals:
            if tg.is_true():
                reward += tg.reward
        done = all(tg.is_true() for tg in self.temp_goals)
        return state, reward, done, info


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
