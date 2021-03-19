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
"""Classes related to reward shaping."""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np
from stable_baselines.common.running_mean_std import RunningMeanStd
from temprl.automata import RewardDFA

from multinav.helpers.gym import State

StateL = State
StateH = State

logger = logging.getLogger(__name__)


class RewardShaper(ABC):
    """Base reward shaper interface.

    A reward shaper computes a reward to sum to the original reward.
    """

    @abstractmethod
    def reset(self, state: State):
        """Reset the class for a new episode.

        :param state: the observation returned by the env.
        """
        pass

    @abstractmethod
    def step(self, state: State, reward: float, done: bool):
        """Compute reward for current step.

        The arguments are the gym retured values.
        :return: the scalar value that should be summed to reward.
        """
        pass


class PotentialRewardShaper(RewardShaper):
    """Classic reward shaping computed on a potential function.

    Some details refer to:
        http://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf
    """

    def __init__(
        self,
        potential_function: Callable[[State], float],
        gamma: float,
        zero_terminal_state: bool,
        normalize_potential: bool = False,
    ):
        """Initialize.

        :param potential_function: callable that computes the potential at
            each state.
        :param gamma: RL discount factor.
        :param zero_terminal_state: if true, the potential of the final states
            is set to zero. See the reference paper.
        :param normalize_potential: if True, the potential function is
            standardized to unit variance and zero mean. Note that these
            moments are estimated while they are produced and the output may
            slowly shift, then settle.
        """
        # Super
        RewardShaper.__init__(self)

        # Store
        self.potential_function = potential_function
        self._gamma = gamma
        self._zero_terminal_state = zero_terminal_state
        self._normalizing = normalize_potential

        # Init
        self._last_potential = 0.0
        self._normalizer = RunningMeanStd(shape=())

    def reset(self, state: State):
        """See super."""
        # Compute potential
        potential = self.potential_function(state)

        # Maybe normalize it
        if self._normalizing:
            potential = self._normalize_potential(potential)

        self._last_potential = potential
        logger.debug(f"Initial state: {state}, potential: {self._last_potential}")

    def step(self, state: State, reward: float, done: bool) -> float:
        """See super."""
        # Compute potentials
        potential = self.potential_function(state)

        if done and self._zero_terminal_state:
            potential = 0

        # Maybe normalize it
        elif self._normalizing:
            potential = self._normalize_potential(potential)

        # Shaping rule
        shaping_reward = self._gamma * potential - self._last_potential

        logger.debug(
            f"State: {state}, potential: {potential}, shaping: {shaping_reward}"
        )

        self._last_potential = potential
        return shaping_reward

    def _normalize_potential(self, potential: float) -> float:
        """Update the moments and normalize the potential.

        Returns a normalized potential according to the current moments.
        Moments are also updated.
        """
        # Update moments
        self._normalizer.update(np.array([potential]))

        # Normalize
        potential = potential - self._normalizer.mean
        potential = potential / np.sqrt(self._normalizer.var + 1e-8)
        return potential


class ValueFunctionRS(PotentialRewardShaper):
    r"""Reward shaping based on value function.

    It takes in input:
    - a value function on the domain of states H.
    - a mapping function of states L -> H.
    The reward shaping applied to l \in L is computed on the value function
    of the corresponding l -> h.
    """

    def __init__(
        self,
        value_function: Callable[[StateH], float],
        mapping_function: Callable[[StateL], StateH],
        gamma: float,
        **kwargs,
    ):
        """
        Initialize the reward shaping wrapper.

        :param value_function: the value function.
        :param mapping_function: the mapping function.
        :param gamma: MDP discount factor.
        :param kwargs: additional base class arguments
        """
        self._value_function = value_function
        self._mapping_function = mapping_function

        # Super
        PotentialRewardShaper.__init__(
            self,
            potential_function=self._potential_function,
            gamma=gamma,
            **kwargs,
        )

    def _potential_function(self, state: StateL) -> float:
        """Definition of the potential function (see class doc)."""
        stateH: StateH = self._mapping_function(state)
        potential = self._value_function(stateH)
        return potential


class AutomatonRS(RewardShaper):
    """Reward shaping based on a DFA.

    It can be applied to environments that have a single temporal goal wrapper
    applied (make sure that reward shaping from TemporalGoal is off).
    The purpose of these rewards is to replace (or be summed to) the original
    reward generated when a temporal goal is complete. It is possible to proof
    that, for a class of temporal goals, these reward induce the same optimal
    policies. The idea is to anticipate the rewards down the path that leads
    to the goal. It assumes that maximizing the policy is equivalent to
    reaching the final states. Potentials are negative, because they
    represent how bad is a state (how far from the goal).

    Better not to apply reward shaping before this, because it can
    interfere with cancel_reward option.

    This is not a proper reward shaping; rather it's just a potential based
    reward function. This doesn't achieve invariance of returns and
    doesn't follow the usual PBRS equation.
    """

    def __init__(
        self,
        goal: RewardDFA,
        rescale: bool = True,
        cancel_reward: bool = False,
    ):
        """Initialize.

        :param goal: an automaton with associated reward.
        :param rescale: if true, the potential function is scaled in [-1, 0]
        :param cancel_reward: if true, cancels the original reward.
            It should still converge to the same optimal policy.
        """
        # Super
        RewardShaper.__init__(self)

        # Init
        self.goal_automaton = goal
        self.__rescale = rescale
        self.__potential_table = self._compute_potential()
        self.__cancel_reward = cancel_reward
        self._last_potential = 0.0

        if self.goal_automaton.reward <= 0:
            logger.warning("The goal automaton has a negative reward associated.")

        # TODO: maybe add a scale factor, so that the sum of rewards is
        #  equal to the temporal goal (even though undiscounted..)

    def _compute_potential(self) -> Dict[int, float]:
        """Compute the potential function from the DFA.

        :return: a dictionary from automaton states to value of the potential
            function.
        """
        # Distance from final states
        distances: Dict[State, int] = self.goal_automaton.levels_to_accepting_states()
        dist_max = max(distances.values())
        d_min = -dist_max - 1

        # Potential
        potential = {
            s: float(-distances[s]) if distances[s] >= 0 else float(d_min)
            for s in self.goal_automaton.states
        }

        if self.__rescale:
            potential = {q: p / abs(d_min) for q, p in potential.items()}

        logger.debug(f"DFA states potential: {potential}")

        return potential

    def potential_function(self, state: State) -> float:
        """Definition of the potential function."""
        assert len(state) == 2, "Expected a tuple of states: (env, automaton)"
        assert len(state[1]) == 1, "Expected only one temporal goal"

        # Tabular lookup
        return self.__potential_table[state[1][0]]

    def reset(self, state: State):
        """See super."""
        self._last_potential = self.potential_function(state)

        logger.debug(f"Initial state: {state}, potential: {self._last_potential}")

    def step(self, state: State, reward: float, done: bool) -> float:
        """See super."""
        # Compute potentials
        potential = self.potential_function(state)
        shaping_reward = potential - self._last_potential
        self._last_potential = potential

        logger.debug(
            f"State: {state}, potential: {potential}, reward: {shaping_reward}"
        )

        # Cancel the original reward; assuming that we've reached the goal
        if done and reward > 0 and self.__cancel_reward:
            if reward != self.goal_automaton.reward:
                raise RuntimeError(
                    "Expected a positive reward for complete temporal goal"
                )
            shaping_reward -= self.goal_automaton.reward
            logger.debug(f"Removed goal reward {self.goal_automaton.reward}")

        return shaping_reward
