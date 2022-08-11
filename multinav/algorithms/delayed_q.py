"""Implementation of Delayed Q learning.

Delayed Q learning, from

@inproceedings{strehl_2006_PACModelfree,
  title = {{{PAC}} Model-Free Reinforcement Learning},
  booktitle = {Proceedings of the 23rd International Conference on {{Machine}} Learning  - {{ICML}} '06},
  author = {Strehl, Alexander L. and Li, Lihong and Wiewiora, Eric and Langford, John and Littman, Michael L.},
  date = {2006},
}
"""

import logging
import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from gym import Env

from multinav.algorithms.q_learning import Learner
from multinav.helpers.gym import discrete_space_size, evaluate
from multinav.wrappers.utils import MyStatsRecorder

logger = logging.getLogger(__name__)

StateT = int
ActionT = int


class DelayedQAgent(Learner):
    """Implementation for Delayed Q-learning algorithm.

    See the paper in module docstring for reference.
    """

    def __init__(
        self,
        env: Env,
        stats_env: MyStatsRecorder,
        gamma: float,
        eps1: float,
        delta: float,
        maxr: float,
        minr: float,
        m: Optional[int] = None,
        rollout_interval: int = 0,
        rollout_episodes: int = 0,
    ):
        """Initialize.

        :param env: environment (discrete state and observations)
        :param stats_envs: the MyStatsRecorder wrapper of env.
        :param gamma: discounting
        :param eps1: a small optimism constant for values (see paper)
        :param delta: guarantee failure probability of the PAC analysis
        :param maxr: maximum reward
        :param minr: minimum reward
        :param m: number of samples per state-action per update (optional).
        :param rollout_interval: minimum number of steps between rollouts.
        :param rollout_episodes: number of episodes for each rollout.
        """
        # Store
        self.env = env
        self.stats_env = stats_env
        self.gamma = gamma
        self.eps1 = eps1
        self.delta = delta
        self.maxr = maxr
        self.minr = minr
        self.rollout_interval = rollout_interval
        self.rollout_episodes = rollout_episodes

        # Constants
        assert self.env.observation_space is not None
        assert self.env.action_space is not None
        self.n_states = discrete_space_size(self.env.observation_space)
        self.n_actions = discrete_space_size(self.env.action_space)

        # Compute constants
        self.m = self._compute_m() if not m else m
        self.maxv = self._compute_maxv()

        # Log
        logger.info(
            f"""DelayedQAgent initialized
            params:
                gamma = %f
                eps1 = %f
                delta = %f
                maxr = %f
                minr = %f

                n_states = %d
                n_actions = %d
                maxv = %f
                m = %d
            """,
            self.gamma,
            self.eps1,
            self.delta,
            self.maxr,
            self.minr,
            self.n_states,
            self.n_actions,
            self.maxv,
            self.m
        )

        # Dict from training step to evaluation metrics
        self.eval_stats: Dict[int, List[List[float]]] = {}

        # Init
        self._init_learner()

    def _init_learner(self):
        """(Re)initialize the learner for the learning loop."""
        self._last_update = 0  # step of last q-value update
        self._last_attempt: DefaultDict[Tuple[StateT, ActionT], int] = (
            defaultdict(Constant(0))
        )  # time of last attempted update
        self._l: DefaultDict[Tuple[StateT, ActionT], int] = (
            defaultdict(Constant(0))
        )  # number of samples for each (s, a)
        self._learn: DefaultDict[Tuple[StateT, ActionT], bool] = (
            defaultdict(Constant(True))
        )  # whether this (s, a) should be updated
        self._u: DefaultDict[Tuple[StateT, ActionT], float] = (
            defaultdict(Constant(0.0))
        )  # empirical value estimate
        self.Q: Dict[StateT, np.ndarray] = (
            defaultdict(NumpyConstant(np.ones(self.n_actions) / (1 - self.gamma)))
        )  # Q-function

    def _compute_m(self) -> int:
        """Compute the number of samples needed for an update."""
        k = 1 / ((1 - self.gamma) * self.eps1)
        n1 = math.log(
            3 * self.n_states * self.n_actions * (
                1 + self.n_states * self.n_actions * k) / self.delta
        )
        n2 = 2 * self.eps1 * self.eps1 * (1 - self.gamma) * (1 - self.gamma)

        return math.ceil(n1 / n2)

    def _compute_maxv(self) -> float:
        """Compute the maximum value."""
        assert self.minr == 0, "Rewards must be positive or null"
        return self.maxr / (1 - self.gamma)

    def learn(self, max_steps: int):
        """Update with Delayed-Q.

        :param max_steps: maximum number of timesteps.
        """
        done = True
        obs = 0  # Any initialization
        should_evaluate = False
        ep = -1

        # Learning loop
        for step in range(max_steps):

            # New episode
            if done:
                if should_evaluate:
                    self.eval_stats[step] = evaluate(
                        env=self.stats_env.env,
                        gamma=self.gamma,
                        policy=lambda state: np.argmax(self.Q[state]),
                        nb_episodes=self.rollout_episodes,
                    )
                should_evaluate = False
                obs = self.env.reset()
                done = False
                ep += 1

            # Step
            action = self._choose_action(obs)
            obs2, reward, done, info = self.env.step(action)

            # Apply
            self._learn_step(
                obs=obs,
                action=action,
                reward=reward,
                obs2=obs2,
                step=step,
            )
            obs = obs2

            # Print
            if ep % 5 == 0:
                print(" Episode:", ep, end="\r")

            # Evaluation
            if self.rollout_interval > 0 and step % self.rollout_interval == 0:
                should_evaluate = True

    def _learn_step(
        self,
        obs: StateT,
        action: ActionT,
        reward: float,
        obs2: StateT,
        step: int,
    ) -> Dict[str, float]:
        """One step of the learning algorithm."""
        # Not learning for this state
        if not self._learn[(obs, action)]:

            # Should we learn next time?
            if self._last_attempt[(obs, action)] < self._last_update:
                self._learn[(obs, action)] = True
                logger.debug(
                    "Learning true for (%s, %d); attempt %d, update %d",
                    obs,
                    action,
                    self._last_attempt[(obs, action)],
                    self._last_update,
                )
            return dict()

        # Update count
        self._l[(obs, action)] += 1
        count = self._l[(obs, action)]

        # Update average value
        val = reward + self.gamma * self.Q[obs2].max()
        self._u[(obs, action)] = self._u[(obs, action)] * (
            (count - 1) / count) + val / count

        # Do not attempt an update yet
        if count < self.m:
            return dict()

        logger.debug("""\
            Attempting update at step %d for (%s, %d):
                old value %f,
                new value %f""",
            step,
            obs,
            action,
            self.Q[obs][action],
            self._u[(obs, action)],
        )

        # Maybe update
        self._last_attempt[(obs, action)] = step
        new_val = self._u[(obs, action)]
        if self.Q[obs][action] - new_val >= 2 * self.eps1:
            self.Q[obs][action] = new_val + self.eps1
            self._last_update = step
            logger.debug("Update succeeded")

        # Maybe stop learning
        elif self._last_attempt[(obs, action)] >= self._last_update:
            self._learn[(obs, action)] = False
            logger.debug("Learning false for %s, %d", obs, action)

        # Clear after attempted updates
        self._u[(obs, action)] = 0.0
        self._l[(obs, action)] = 0

        return dict()

    def _choose_action(self, obs: StateT):
        """Select an action.

        The action is greedy with respect to learner Q-function.
        """
        return np.argmax(self.Q[obs]).item()


class Constant:
    """Class that generates a constant when called.

    Unfortunately, lambdas can't be pickled.
    """

    def __init__(self, const):
        """Store value."""
        self.const = const

    def __call__(self):
        """Return constant."""
        return self.const


class NumpyConstant(Constant):
    """Generate new arrays."""

    def __call__(self):
        """Return new array."""
        return np.array(self.const)
