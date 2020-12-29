"""Helper functions and classes related to the stable_baselines library."""

import json
import os
import pickle

from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback


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

    def __init__(self, save_path, save_freq=None, name_prefix="model", extra=None):
        """Initialize.

        :param save_path: model checkpoints path.
        :param save_freq: number of steps between each save (None means never).
        :param name_prefix: just a name for the saved weights.
        :param extra: an optional, additional pickable object that is saved with
            the main model.
        """
        BaseCallback.__init__(self)

        # Store
        self._save_freq = save_freq
        self._counters_file = os.path.join(save_path, os.path.pardir, "counters.json")
        self._chkpt_format = os.path.join(save_path, name_prefix + "_{step}")
        self._chkpt_extension = ".zip"
        self.extra_model = extra
        self._extra_format = os.path.join(save_path, "Extra_{step}.pickle")

    def _update_counters(self, filepath, step, extra_file=None):
        """Update the file of counters with a new entry.

        :param filepath: checkpoint that is being saved
        :param step: current global step
        :param extra_file: associated extra saved model
        """
        counters = {}

        # Load
        if os.path.exists(self._counters_file):
            with open(self._counters_file) as f:
                counters = json.load(f)

        filepath = os.path.relpath(filepath)
        counters[filepath] = dict(step=step, extra=extra_file)

        # Save
        with open(self._counters_file, "w") as f:
            json.dump(counters, f, indent=4)

    def save(self, step):
        """Manually save a checkpoint.

        :param step: the current step of the training
            (used just to identify checkpoints).
        """
        # Save model
        model_path = self._chkpt_format.format(step=step)
        self.model.save(model_path)
        # Save extra
        if self.extra_model is not None:
            extra_path = self._extra_format.format(step=step)
            with open(extra_path, "wb") as f:
                pickle.dump(self.extra_model, f)
        else:
            extra_path = None

        self._update_counters(
            filepath=model_path + self._chkpt_extension,
            step=step,
            extra_file=extra_path,
        )

    def load(self, path):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path.
        :return: the model, the extra object (can be None) and associated
            counters.
        """
        # Restore
        path = os.path.relpath(path)
        model = DQN.load(load_path=path)
        print("> Loaded:", path)

        # Read counters
        with open(self._counters_file) as f:
            data = json.load(f)
        counters = data[path]

        # Load extra
        extra_path = counters.pop("extra")
        if extra_path:
            with open(extra_path, "rb") as f:
                extra_model = pickle.load(f)
        else:
            extra_model = None

        return model, extra_model, counters

    def _on_step(self):
        """Automatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)
