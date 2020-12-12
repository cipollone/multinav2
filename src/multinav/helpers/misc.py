"""Misc helpers."""

import multiprocessing
import os
import random
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

from multinav.helpers.gym import MyStatsRecorder

# Output base dir
output_base = "outputs"


def set_seed(env: gym.Env, seed: int = None):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)


@dataclass()
class Stats:
    """Record RL experiment statistics."""

    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    total_steps: int = 0
    timestamps: List[int] = field(default_factory=list)


def stats_from_env(env: MyStatsRecorder) -> Stats:
    """Get statistics from environment."""
    return Stats(
        episode_lengths=env.episode_lengths,
        episode_rewards=env.episode_rewards,
        total_steps=env.total_steps,
        timestamps=env.timestamps,
    )


def plot_stats(stats: Stats):
    """Plot the stats."""
    plt.title("Episode lengths")
    plt.plot(stats.episode_lengths)
    plt.show()

    plt.title("Episode Rewards")
    plt.plot(stats.episode_rewards)
    plt.show()


def plot_average_stats(
    stats_list: Sequence[Sequence[Stats]],
    labels: Sequence[str],
    show: bool = True,
    prefix: str = "",
):
    """Plot average stats."""
    assert len(stats_list) == len(
        labels
    ), "Please provide the correct number of labels."
    attributes = ["episode_lengths", "episode_rewards"]
    figures = []
    for attribute in attributes:
        f = plt.figure()
        ax = f.add_subplot()
        ax.set_title(attribute)
        for label, history_list in zip(labels, stats_list):
            _plot_stats_attribute(history_list, attribute, label, ax=ax)
        figures.append(f)
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(prefix, attribute + ".svg"))

    return figures


def _plot_stats_attribute(stats_list: Sequence[Stats], attribute: str, label, ax=None):
    """Plot a certain attribute of a collection of histories."""
    data = np.asarray([getattr(h, attribute) for h in stats_list])
    df = DataFrame(data.T)

    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    sns_ax = sns.lineplot(df_mean.index, df_mean, label=label, ax=ax)
    sns_ax.fill_between(df_mean.index, df_mean - df_std, df_mean + df_std, alpha=0.3)


class Experiment:
    """Class to coordinate a RL experiment."""

    def __init__(
        self,
        env: gym.Env,
        rl_algorithm: Callable[[gym.Env, Any], Any],
        algorithm_params: Dict,
        nb_runs: int = 50,
        nb_processes: int = 8,
        seeds: Sequence[int] = (),
    ):
        """
        Set up the experiment.

        :param env: the environment.
        :param rl_algorithm: the RL algorithm to use.
                 It is a callable that expects the environment
                 and any collection of parameters
        :param nb_runs: the number of runs to run in parallel.
        :param nb_processes: the number of processes available.
        :param seeds: the random seeds to set up a run.
        """
        self.env = env
        self.rl_algorithm = rl_algorithm
        self.algorithm_params = algorithm_params
        self.nb_runs = nb_runs
        self.nb_processes = nb_processes
        self.seeds = list(seeds) if len(seeds) == nb_runs else list(range(nb_runs))

    @staticmethod
    def _do_job(env, algorithm, seed, **params):
        """
        Do a single run.

        We put it here so it is pickable.
        """
        set_seed(env, seed)
        Q = algorithm(env, **params)
        stats = stats_from_env(env)
        return Q, stats

    def run(self) -> List[Stats]:
        """
        Run the experiments.

        :return: a list of statistics, one for each run.
        """
        pool = multiprocessing.Pool(processes=self.nb_processes)

        results = [
            pool.apply_async(
                self._do_job,
                args=(self.env, self.rl_algorithm, seed),
                kwds=self.algorithm_params,
            )
            for seed in self.seeds
        ]

        try:
            for p in results:
                p.wait()
        except KeyboardInterrupt:
            pass

        stats = []
        for p in filter(lambda x: x.ready(), results):
            _, stat = p.get()
            stats.append(stat)
        return stats


def prepare_directories(  # noqa: ignore
    env_name, resuming=False, args=None, no_create=False
):
    """Prepare the directories where weights and logs are saved.

    Just to know the output paths, call this function with `no_create=True`.

    :param env_name: the actual paths are a composition of `output_base` and
        `env_name`.
    :param resuming: if true, the directories are not deleted.
    :param args: argsparse.Namespace of arguments. If given, this is saved
        to 'args' file inside the log directory.
    :param no_create: do not touch the files; just return the current paths
        (implies resuming).
    :return: two paths, respectively for models and logs.
    """
    if no_create:
        resuming = True

    # Choose diretories
    models_path = os.path.join(output_base, env_name, "models")
    logs_path = os.path.join(output_base, env_name, "logs")
    dirs = (models_path, logs_path)

    # Delete old ones
    if not resuming:
        if any(os.path.exists(d) for d in dirs):
            print("Old logs and models will be deleted. Continue (Y/n)? ", end="")
            c = input()
            if c not in ("y", "Y", ""):
                quit()

        # New
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    # Logs and models for the same run are saved in
    #   directories with increasing numbers
    i = 0
    while os.path.exists(os.path.join(logs_path, str(i))) or os.path.exists(
        os.path.join(models_path, str(i))
    ):
        i += 1

    # Should i return the current?
    if no_create:
        last_model_path = os.path.join(models_path, str(i - 1))
        last_log_path = os.path.join(logs_path, str(i - 1))
        if (
            i == 0
            or not os.path.exists(last_log_path)
            or not os.path.exists(last_model_path)
        ):
            raise RuntimeError("Dirs should be created first")
        return (last_model_path, last_log_path)

    # New dirs
    model_path = os.path.join(models_path, str(i))
    log_path = os.path.join(logs_path, str(i))
    os.mkdir(model_path)
    os.mkdir(log_path)

    # Save arguments
    if args is not None:
        raise NotImplementedError("Feature still not used.")

    return (model_path, log_path)
