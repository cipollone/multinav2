"""Implementation of Delayed Q learning.

Delayed Q learning, from

@inproceedings{strehl_2006_PACModelfree,
  title = {{{PAC}} Model-Free Reinforcement Learning},
  booktitle = {Proceedings of the 23rd International Conference on {{Machine}} Learning  - {{ICML}} '06},
  author = {Strehl, Alexander L. and Li, Lihong and Wiewiora, Eric and Langford, John and Littman, Michael L.},
  date = {2006},
}
"""

import math
from collections import defaultdict
from typing import DefaultDict, Dict, Tuple

import numpy as np
from gym import Env
from gym.spaces import Discrete

StateT = int
ActionT = int


class DelayedQAgent:
    """Implementation for Delayed Q-learning algorithm.

    See the paper in module docstring for reference.
    """

    def __init__(
        self,
        env: Env,
        gamma: float,
        eps1: float,
        delta: float,
        maxr: float,
        minr: float,
    ):
        """Initialize.

        :param env: environment (discrete state and observations)
        :param gamma: discounting
        :param eps1: a small optimism constant for values (see paper)
        :param delta: guarantee failure probability of the PAC analysis
        :param maxr: maximum reward
        :param minr: minimum reward
        """
        # Store
        self.env = env
        self.gamma = gamma
        self.eps1 = eps1
        self.delta = delta
        self.maxr = maxr
        self.minr = minr

        # Constants
        assert isinstance(self.env.observation_space, Discrete)
        assert isinstance(self.env.action_space, Discrete)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # Compute constants
        self.m = self._compute_m()
        self.maxv = self._compute_maxv()

    def _init_learner(self):
        """(Re)initialize the learner for the learning loop."""
        self._last_update = 0  # step of last q-value update
        self._last_attempt: DefaultDict[Tuple[StateT, ActionT], int] = (
            defaultdict(lambda: 0)
        )  # time of last attempted update
        self._l: DefaultDict[Tuple[StateT, ActionT], int] = (
            defaultdict(lambda: 0)
        )  # number of samples for each (s, a)
        self._learn: DefaultDict[Tuple[StateT, ActionT], bool] = (
            defaultdict(lambda: True)
        )  # whether this (s, a) should be updated
        self._u: DefaultDict[Tuple[StateT, ActionT], float] = (
            defaultdict(lambda: 0.0)
        )  # empirical value estimate
        self.Q: Dict[StateT, np.ndarray] = (
            defaultdict(lambda: np.ones(self.n_actions) / (1 - self.gamma))
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

        # Learning loop
        for step in range(max_steps):

            # New episode
            if done:
                obs = self.env.reset()

            # Step
            action = self._choose_action(obs)
            obs2, reward, done, info = self.env.step(action)

            # Apply
            stats = self._learn_step(
                obs=obs,
                action=action,
                reward=reward,
                obs2=obs2,
                step=step,
            )

            # TODO: log stats
            # TODO: active,passive

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
            return dict()

        # Update count
        count = self._l[(obs, action)]
        self._l[(obs, action)] = count + 1

        # Update average value
        val = reward + self.gamma * self.Q[obs2].max()
        self._u[(obs, action)] = self._u[(obs, action)] * (
            (count - 1) / count) + val / count

        # Do not attempt an update yet
        if count + 1 < self.m:
            return dict()

        # Maybe update
        new_val = self._u[(obs, action)]
        if self.Q[obs][action] - new_val >= 2 * self.eps1:
            self.Q[obs][action] = new_val + self.eps1
            self._last_update = step

        # Maybe stop learning
        elif self._last_attempt[(obs, action)] >= self._last_update:
            self._learn[(obs, action)] = False

        # Clear after attempted updates
        self._last_attempt[(obs, action)] = 0
        self._u[(obs, action)] = 0.0
        self._l[(obs, action)] = 0

        return dict()

    def _choose_action(self, obs: StateT):
        """Select an action.

        The action is greedy with respect to learner Q-function.
        """
        return np.argmax(self.Q[obs]).item()
