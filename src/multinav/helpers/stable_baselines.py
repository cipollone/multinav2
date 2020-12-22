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

    This class manages checkpoints, save and restore.
    It also act as callback class so that it can be used inside
    stable_baselines learning loop.
    If you don't plan to use it as callback, assign the model to self.model.
    """

    # TODO: referring specifically to the normalizer it's ugly. Generic pickable

    def __init__(self, save_path, normalizer, save_freq=None, name_prefix="model"):
        """Initialize.

        :param save_path: model checkpoints path.
        :param normalizer: a VecNormalize instance.
        :param save_freq: number of steps between each save (None means never).
        :param name_prefix: just a name for the saved weights.
        """
        BaseCallback.__init__(self)

        # Store
        self.normalizer_model = normalizer
        self._save_freq = save_freq
        self._counters_file = os.path.join(save_path, os.path.pardir, "counters.json")
        self._chkpt_format = os.path.join(save_path, name_prefix + "_{step}")
        self._chkpt_extension = ".zip"
        self._normalizer_format = os.path.join(save_path, "VecNormalize_{step}.pickle")

    def _update_counters(self, filepath, step, normalizer_file):
        """Update the file of counters with a new entry.

        :param filepath: checkpoint that is being saved
        :param step: current global step
        :param normalizer_file: associated normalizer
        """
        counters = {}

        # Load
        if os.path.exists(self._counters_file):
            with open(self._counters_file) as f:
                counters = json.load(f)

        filepath = os.path.relpath(filepath)
        counters[filepath] = dict(step=step, normalizer=normalizer_file)

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
        # Save checkpoint
        normalizer_path = self._normalizer_format.format(step=step)
        with open(normalizer_path, "wb") as f:
            pickle.dump(self.normalizer_model, f)

        self._update_counters(
            filepath=model_path + self._chkpt_extension,
            step=step,
            normalizer_file=normalizer_path,
        )

    def load(self, path):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path.
        :return: the model and associated counters.
        """
        # Restore
        path = os.path.relpath(path)
        model = DQN.load(load_path=path)
        print("> Loaded:", path)

        # Read counters
        with open(self._counters_file) as f:
            data = json.load(f)
        counters = data[path]

        # Restore normalizer
        normalizer_path = counters.pop("normalizer")
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)

        return model, normalizer, counters

    def _on_step(self):
        """Automatic save."""
        if self._save_freq is None:
            return
        if self.num_timesteps % self._save_freq == 0:
            self.save(step=self.num_timesteps)
