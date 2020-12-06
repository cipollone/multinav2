"""Helpers related to TempRL wrappers."""
import numpy as np
from temprl.wrapper import TemporalGoalWrapper

from multinav.helpers.notebooks import automaton_to_rgb


class MyTemporalGoalWrapper(TemporalGoalWrapper):
    """
    Custom version of TemporalGoalWrapper.

    In particular, it changes the rendering, by concatenating
    the frame of the environment and the frame of the automata.
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
