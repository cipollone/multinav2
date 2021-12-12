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
"""Q-Learning implementation."""
import logging
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence

import gym
import numpy as np

from multinav.wrappers.utils import MyStatsRecorder

logger = logging.getLogger(__name__)

State = Any
QTableType = Dict[State, np.ndarray]


class QLearning:
    """Learn a Q-function from a Gym env using vanilla Q-Learning."""

    def __init__(
        self,
        env: gym.Env,
        stats_envs: Sequence[MyStatsRecorder],
        seed: int,
        total_timesteps: int = 2000000,
        alpha: float = 0.1,
        eps: float = 0.1,
        gamma: float = 0.9,
        learning_rate_decay: bool = False,
        learning_rate_end: float = 0.0,
        epsilon_decay: bool = False,
        epsilon_end: float = 0.0,
        action_bias: Optional[QTableType] = None,
        action_bias_eps: float = 0.0,
        exploration_policy: Optional[QTableType] = None,
        initial_Q: Optional[QTableType] = None,
        active_passive_agents: bool = False,
        passive_reward_getter: Optional[Callable[[], float]] = None,
        initial_passive_Q: Optional[QTableType] = None,
    ):
        """Initialize.

        :param env: an environment with discrete actions and observations.
        :param stats_envs: a list of MyStatsRecorder wrappers.
            Just one is usually enough, but if active_passive_agents is True,
            a second one is needed for the passive agent.
        :param total_timesteps: total number of optimizations.
        :param alpha: learning rate.
        :param eps: probability of random action.
        :param gamma: RL discount factor.
        :param learning_rate_decay: whether the learning rate should linearly
            decrease to learning_rate_end.
        :param learning_rate_end: last value of alpha.
        :param epsilon_decay: whether epsilon should decay to epsilon_end.
        :param epsilon_end: last value of epsilon.
        :param action_bias: A Q function that is used just as bias to select
            the actions during exploration. If you find this too deterministic
            it's possible to use action_bias_eps.
        :param action_bias_eps: During exploration, with eps probability
            select an action with uniform distribution, idependently of
            action_bias.
        :param exploration_policy: if given, this Q-table is used to select
            greedy actions instead of this learner's Q-table. Epsilon greedy still
            applies, but action bias is ignored.
        :param initial_Q: the initial Q function.
        :param active_passive_agents: if True, a second agent is instantiated.
            This does not execute any action, but only learns from what
            the active agent does. For the same observations, it can learn
            on a different set of rewards, given by passive_reward_getter.
        :param passive_reward_getter: a callable that returns the last reward
            that should be passed to the passive agent.
        :param initial_passive_Q: initialization for the passive agent.
        :return the Q function: a dictionary from states to array of Q values
            for every action.
        """
        # Check
        if active_passive_agents and not passive_reward_getter:
            raise TypeError(
                "passive_reward_getter is mandatory when training " "a passive agent."
            )
        if initial_Q and active_passive_agents and not initial_passive_Q:
            raise TypeError(
                "You must supply an initialization for the passive agent too."
            )
        assert (not active_passive_agents and len(stats_envs) == 1) or (
            active_passive_agents and len(stats_envs) == 2
        ), "You should pass one stats environment for each agent"

        # Store
        self.env = env
        self.stats_envs = stats_envs
        self.total_timesteps = total_timesteps
        self.alpha0 = alpha
        self.eps0 = eps
        self.gamma = gamma
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_end = learning_rate_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.action_bias = action_bias
        self.action_bias_eps = action_bias_eps
        self.exploration_policy = exploration_policy
        self.active_passive_agents = active_passive_agents
        self.passive_reward_getter = passive_reward_getter

        # Initialize
        self.__rng = np.random.default_rng(seed)
        self.nb_actions = env.action_space.n

        initialization_fn = partial(
            self._Q_initialization_fn, self.__rng, self.nb_actions
        )

        # New agents
        self.Q: QTableType = defaultdict(initialization_fn)
        self.passive_Q: QTableType = defaultdict(initialization_fn)

        # Maybe restore
        if initial_Q is not None:
            self.Q = (
                initial_Q
                if isinstance(initial_Q, defaultdict)
                else defaultdict(initialization_fn, initial_Q)
            )
            # Same for passive agent
            if active_passive_agents:
                assert initial_passive_Q is not None  # mypy fix
                self.passive_Q = (
                    initial_passive_Q
                    if isinstance(initial_passive_Q, defaultdict)
                    else defaultdict(initialization_fn, initial_passive_Q)
                )

        # Moving vars
        self.alpha = self.alpha0
        self.eps = self.eps0

    def learn(self) -> QTableType:
        """Start training.

        The result is stored in self.Q.
        :return: The q table stored in self.Q.
        """
        done = True
        state = None  # for type checks
        for step in range(self.total_timesteps):

            if done:
                state = self.env.reset()
                done = False

            # Step
            action = self._choose_action(state)
            state2, reward, done, _info = self.env.step(action)

            # Apply
            td_update = self._optimize_q_step(
                Q=self.Q,
                state=state,
                action=action,
                reward=reward,
                state2=state2,
                gamma=self.gamma,
                alpha=self.alpha,
            )

            # Apply also for passive agent
            td_passive_update = 0.0
            if self.active_passive_agents:
                assert self.passive_reward_getter is not None  # mypy fix
                td_passive_update = self._optimize_q_step(
                    Q=self.passive_Q,
                    state=state,
                    action=action,
                    reward=self.passive_reward_getter(),
                    state2=state2,
                    gamma=self.gamma,
                    alpha=self.alpha,
                )

            state = state2

            # Decays
            if step % 20 == 0:
                self.update_decays(step)
                print(" Eps:", round(self.eps, 3), end="\r")

            # Log
            self.stats_envs[0].update_extras(td=abs(td_update))
            if self.active_passive_agents:
                self.stats_envs[1].update_extras(td=abs(td_passive_update))

        print()
        return self.Q

    def update_decays(self, step: int):
        """Update the value subject to decay.

        :param step: current step.
        """
        frac = step / self.total_timesteps
        if self.learning_rate_decay:
            self.alpha = self.alpha0 * (1 - frac) + self.learning_rate_end * frac
        if self.epsilon_decay:
            self.eps = self.eps0 * (1 - frac) + self.epsilon_end * frac

    @staticmethod
    def _optimize_q_step(
        Q: QTableType,
        state: State,
        action: int,
        reward: float,
        state2: State,
        gamma: float,
        alpha: float,
    ) -> float:
        """Apply a single optimization of the Q table.

        This is separated from the main loop to leave space for future
        extensions (two Qs'?).
        """
        # Compute update
        td_update = reward + gamma * np.max(Q[state2]) - Q[state][action]

        logger.debug(
            f"Q[{state}][{action}] = {Q[state][action]} "
            f"-> {reward + gamma * np.max(Q[state2])}"
        )

        # Apply
        Q[state][action] += alpha * td_update

        return td_update

    def _choose_action(self, state: State) -> int:
        """Select an action.

        If action_bias is None, the policy is epsilon greedy.
        Random actions are uniform, if action_bias is None, biased otherwise.
        Biased actions means that they are chosed from action_bias, with
        1-action_bias_eps probability.
        If an exploration policy is given, greedy actions are selected
        according to this one.
        """
        # Exploration
        if self.__rng.random() < self.eps:
            bias_sample = self.__rng.random()

            # Uniform probability
            if self.action_bias is None or bias_sample < self.action_bias_eps:
                return self.__rng.integers(self.nb_actions)

            # Biased action
            else:
                return np.argmax(self.action_bias[state]).item()

        # Greedy
        else:
            # Exploration policy?
            if self.exploration_policy:
                return np.argmax(self.exploration_policy[state]).item()

            # Learner Q function
            else:
                return np.argmax(self.Q[state]).item()

    @staticmethod
    def _Q_initialization_fn(rng, nb_actions) -> np.ndarray:
        """Initialize values for the Q function.

        This is static, because it will be pickled by the agent.
        :param rng: numpy random number generator.
        :param nb_actions: number of actions.
        :return: a row of initial values.
        """
        return rng.standard_normal(nb_actions) * sys.float_info.epsilon
