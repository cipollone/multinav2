"""Helper functions and classes related to the stable_baselines library."""
import os
from typing import Optional

from stable_baselines import DQN
from stable_baselines.common import BaseRLModel
from stable_baselines.common.callbacks import BaseCallback

from multinav.helpers.misc import Saver
from multinav.wrappers.utils import MyStatsRecorder


class RendererCallback(BaseCallback):
    """Just render at each frame."""

    def _on_step(self):
        """Do it."""
        self.training_env.render()


class CustomCheckpointCallback(BaseCallback):
    """Manage model checkpoints.

    This class manages checkpoints, save and restore. It also act as callback
    class so that it can be used inside stable_baselines learning loop. If you
    don't plan to use it as callback, assign the model to self.model by
    yourself before saving.
    """

    def __init__(
        self,
        save_path: str,
        save_freq: Optional[int] = None,
        name_prefix: str = "model",
        extra=None,
    ):
        """Initialize.

        :param save_path: model checkpoints path.
        :param save_freq: number of steps between each save (None means never).
        :param name_prefix: just a name for the saved weights.
        :param extra: an optional, additional pickable object that is saved with
            the main model.
        """
        # Super
        BaseCallback.__init__(self)

        # Store
        self._save_freq = save_freq
        self.saver = Saver(
            saver=lambda path: None,  # This will be set in init_callback
            loader=DQN.load,
            save_path=save_path,
            name_prefix=name_prefix,
            model_ext=".zip",
            extra=extra,
        )

    def init_callback(self, model: BaseRLModel) -> None:
        """Initialize vars; stable_baselines interface."""
        self.saver.saver = model.save
        BaseCallback.init_callback(self, model=model)

    def save(self, step):
        """Save a checkpoint; @see Saver."""
        self.saver.save(step=step)

    def load(self, path):
        """Load from a checkpoint; @see Saver."""
        return self.saver.load(path=path)

    def _on_step(self):
        """Automatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)


class StatsLoggerCallback(BaseCallback):
    """Save statistics collected by MyStatsRecorder to a file."""

    def __init__(
        self,
        stats_recorder: MyStatsRecorder,
        log_path: str,
        log_freq: int,
    ):
        """Initialize.

        :param stats_recorder: a MyStatsRecorder that wraps the env in use.
        :param log_path: directory where to save logs.
        :param log_freq: number of steps between each save.
        """
        # Super
        BaseCallback.__init__(self)

        # Store
        self._stats_recorder = stats_recorder
        self._log_file = os.path.join(log_path, "log_table.txt")
        self._log_freq = log_freq
        self._log_properties = [
            "episode_lengths",
            "episode_returns",
            "episode_td_max",
        ]
        self.__file_created = False

    def log(self):
        """Save the current statistics."""
        # TODO: tensorboard instead? or also save plot
        # Initialize file
        if not self.__file_created:
            header = ", ".join(self._log_properties) + "\n"
            with open(self._log_file, "w") as f:
                f.write(header)

        # Timesteps
        a_trace = getattr(self._stats_recorder, self._log_properties[0])
        n_samples = len(a_trace)

        # Group by timestep, not property
        by_timestep = [
            [getattr(self._stats_recorder, name)[i]
                for name in self._log_properties]
            for i in range(n_samples)
        ]

        # String representation
        lines = [
            ", ".join([str(x) for x in values]) + "\n"
            for values in by_timestep
        ]

        # Save
        with open(self._log_file, "a") as f:
            f.writelines(lines)

    def _on_step(self):
        """Save to file sometimes."""
        if self.num_timesteps % self._log_freq == 0:
            self.log()
