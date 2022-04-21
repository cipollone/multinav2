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
from typing import DefaultDict, Tuple

from gym import Env
from gym.spaces import Discrete
from gym.wrappers import TransformReward

from multinav.helpers.gym import find_wrapper

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

        # Compute constants
        self.m = self._compute_m()
        self.maxv = self._compute_maxv()

        # Initializations
        self.last_t = 0  # step of last q-value update
        self.l: DefaultDict[Tuple[StateT, ActionT], int] = (
            defaultdict(lambda: 0)
        )  # number of samples for each (s, a)
        self.u: DefaultDict[Tuple[StateT, ActionT], float] = (
            defaultdict(lambda: 0.0)
        )  # empirical value estimate
        self.Q: DefaultDict[Tuple[StateT, ActionT], float] = (
            defaultdict(lambda: self.maxv)
        )  # Q-function

    def _compute_m(self) -> int:
        """Compute the number of samples needed for an update."""
        # Check
        assert isinstance(self.env.observation_space, Discrete)
        assert isinstance(self.env.action_space, Discrete)

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n

        k = 1 / ((1 - self.gamma) * self.eps1)
        n1 = math.log(
            3 * n_states * n_actions * (
                1 + n_states * n_actions * k) / self.delta
        )
        n2 = 2 * self.eps1 * self.eps1 * (1 - self.gamma) * (1 - self.gamma)

        return math.ceil(n1 / n2)

    def _compute_maxv(self) -> float:
        """Compute the maximum value."""
        assert self.minr == 0, "Rewards must be positive or null"
        return self.maxr / (1 - self.gamma)
