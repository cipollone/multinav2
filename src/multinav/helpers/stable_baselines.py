"""Helper functions and classes related to the stable_baselines library."""

from typing import Optional

from stable_baselines import DQN
from stable_baselines.common import BaseRLModel
from stable_baselines.common.callbacks import BaseCallback

from multinav.helpers.misc import Saver


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
        self._saver = Saver(
            model=None,  # type: ignore
            loader=DQN.load,
            save_path=save_path,
            name_prefix=name_prefix,
            model_ext=".zip",
            extra=extra,
        )

    def init_callback(self, model: BaseRLModel) -> None:
        """Initialize vars; stable_baselines interface."""
        self._saver.model = model
        BaseCallback.init_callback(self, model=model)

    def save(self, step):
        """Save a checkpoint; @see Saver."""
        self._saver.save(step=step)

    def load(self, path):
        """Load from a checkpoint; @see Saver."""
        return self._saver.load(path=path)

    def _on_step(self):
        """Automatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)
