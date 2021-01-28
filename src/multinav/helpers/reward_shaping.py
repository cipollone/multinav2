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

from pythomata.dfa import DFA

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
    ):
        """Initialize.

        :param potential_function: callable that computes the potential at
            each state.
        :param gamma: RL discount factor.
        :param zero_terminal_state: if true, the potential of the final states
            is set to zero. See the reference paper.
        """
        RewardShaper.__init__(self)

        self.potential_function = potential_function
        self._gamma = gamma
        self._zero_terminal_state = zero_terminal_state

        self._last_state: State = None

    def reset(self, state: State):
        """See super."""
        self._last_state = state

    def step(self, state: State, reward: float, done: bool) -> float:
        """See super."""
        # Compute potentials
        v1 = self.potential_function(self._last_state)
        v2 = self.potential_function(state)
        if done and self._zero_terminal_state:
            v2 = 0

        shaping_reward = self._gamma * v2 - v1

        logger.debug(f"State: {state}, value: {v2}, shaping: {shaping_reward}")

        self._last_state = state
        return shaping_reward


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
        zero_terminal_state: bool,
    ):
        """
        Initialize the reward shaping wrapper.

        :param value_function: the value function.
        :param mapping_function: the mapping function.
        :param gamma: MDP discount factor.
        :param zero_terminal_state: if the terminal state of
          a trajectory should have potential equal to zero. See super.
        """
        self._value_function = value_function
        self._mapping_function = mapping_function

        # Super
        PotentialRewardShaper.__init__(
            self,
            potential_function=self._potential_function,
            gamma=gamma,
            zero_terminal_state=zero_terminal_state,
        )

    def _potential_function(self, state: StateL) -> float:
        """Definition of the potential function (see class doc)."""
        stateH: StateH = self._mapping_function(state)
        potential = self._value_function(stateH)
        return potential


class AutomatonRS(PotentialRewardShaper):
    """Reward shaping based on a DFA.

    It can be applied to environments that have a single temporal goal.
    The purpose of these rewards is to replace (or be summed to) the original
    reward generated when a temporal goal is complete. It is possible to proof
    that, for a class of temporal goals, these reward induce the same optimal
    policies. The idea is to anticipate the rewards down the path that leads
    to the goal. It assumes that maximizing the policy is equivalent to
    reaching the final states. Rewards generated are negative, because they
    represent how bad is a state (how far from the goal).
    """

    def __init__(self, dfa: DFA, gamma: float, rescale: bool = True):
        """Initialize.

        :param dfa: the reward automaton. The reward associated to this
            temporal goal must be positive.
        :param gamma: RL discount factor.
        :param rescale: if true, the potential function is scaled in [-1, 0]
        """
        # Init
        self.__dfa = dfa
        self.__rescale = rescale
        self.__potential_table = self._compute_potential()

        # Super
        PotentialRewardShaper.__init__(
            self,
            potential_function=self._potential_function,
            gamma=gamma,
            zero_terminal_state=False,  # Policy invariance guaranteed
        )

    def _compute_potential(self) -> Dict[int, float]:
        """Compute the potential function from the DFA.

        :return: a dictionary from automaton states to value of the potential
            function.
        """
        # Distance from final states
        distances: Dict[State, int] = self.__dfa.levels_to_accepting_states()
        dist_max = max(distances.values())
        d_min = -dist_max - 1

        # Potential
        potential = {
            s: float(-distances[s]) if distances[s] >= 0 else float(d_min)
            for s in self.__dfa.states
        }

        logger.debug(f"DFA states potential: {potential}")

        if self.__rescale:
            potential = {q: p / d_min for q, p in potential.items()}

        return potential

    def _potential_function(self, state: State) -> float:
        """Definition of the potential function."""
        assert len(state) == 2, "Expected a tuple of states: (env, automaton)"
        assert state[1] in self.__dfa.states

        # Tabular lookup
        return self.__potential_table[state[1]]
