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
"""This module contains helpers related to OpenAI Gym."""
import itertools
import random
from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
from graphviz import Digraph
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.wrappers import TimeLimit

State = Any
Action = int
Probability = float
Reward = float
Done = bool
Transition = Tuple[Probability, State, Reward, Done]
Transitions = Dict[State, Dict[Action, List[Transition]]]
# Other aliases
StateL = State
StateH = State


def from_discrete_env_to_graphviz(
    env: "MyDiscreteEnv",
    state2str: Callable[[int], str] = lambda s: str(s),
    action2str: Callable[[int], str] = lambda a: str(a),
) -> Digraph:
    """From discrete environment to graphviz."""
    g = Digraph()
    g.attr(rankdir="LR")
    for state in range(env.nS):
        state_str = state2str(state)
        g.node(state_str)
        for (action, transitions) in env.P.get(state, {}).items():
            action_str = action2str(action)
            for (prob, next_state, reward, done) in transitions:
                if np.isclose(prob, 0.0):
                    continue
                taken_transition = False
                if (
                    env.laststate == state
                    and env.lastaction == action
                    and env.s == next_state
                ):
                    taken_transition = True
                next_state_str = state2str(next_state)
                g.edge(
                    state_str,
                    next_state_str,
                    label=f"{action_str}, p={prob}, r={reward}, done={done}",
                    color="red" if taken_transition else None,
                )

    if env.laststate is not None:
        g.node(state2str(env.laststate), fillcolor="lightyellow", style="filled")
    g.node(state2str(env.s), fillcolor="lightsalmon", style="filled")
    return g


class MyDiscreteEnv(DiscreteEnv):
    """
    A custom version of DicreteEnv.

    Like DiscreteEnv, but adds:

    - 'laststate' for rendering purposes
    - 'available_actions' to get the available action from a state.
    - 'raise ValueError if action is not available in the current state.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the environment."""
        super().__init__(*args, **kwargs)

        self.laststate = None

    def reset(self):
        """Reset the enviornment."""
        self.laststate = None
        return super().reset()

    def step(self, a):
        """Do a step in the enviornment."""
        self.laststate = deepcopy(self.s)
        if a not in self.available_actions(self.s):
            raise ValueError(f"Cannot perform action {a} in state {self.s}.")
        return super().step(a)

    def _is_legal_state(self, state: State):
        """Check that it is a legal state."""
        assert state in self.P, f"{state} is not a legal state."

    def _is_legal_action(self, action: Action):
        """Check that it is a legal action."""
        assert 0 <= action < self.nA, f"{action} is not a legal action."

    def available_actions(self, state):
        """Get the available action from a state."""
        self._is_legal_state(state)
        return set(self.P[state].keys())


def _random_action(env: gym.Env, state):
    available_actions = getattr(
        env, "available_actions", lambda _: list(iter_space(env.action_space))
    )
    actions = available_actions(state)
    return random.choice(list(actions))


def rollout(
    env: gym.Env,
    nb_episodes: int = 1,
    max_steps: Optional[int] = None,
    policy=lambda env, state: _random_action(env, state),
    callback=lambda env, step: None,
):
    """
    Do a rollout.

    :param env: the OpenAI Gym environment.
    :param nb_episodes: the number of rollout episodes.
    :param max_steps: maximum number of steps per episodes.
    :param policy: a callable that takes the enviornment and the state and returns the action.
    :param callback: a callback that takes the environment and it is called at each step.
    :return: None
    """
    if max_steps:
        env = TimeLimit(env, max_episode_steps=max_steps)
    for _ in range(nb_episodes):
        state = env.reset()
        done = False
        callback(env, (state, 0.0, done, {}))
        while not done:
            action = policy(env, state)
            state, reward, done, info = env.step(action)
            callback(env, (state, reward, done, info))


@singledispatch
def iter_space(_):
    """Iterate over a Gym space."""
    raise NotImplementedError


@iter_space.register(Discrete)
def _(space: Discrete):  # type: ignore
    """Iterate over a discrete state space."""
    for i in range(space.n):
        yield i


@iter_space.register(MultiDiscrete)  # type: ignore
def _(space: MultiDiscrete):
    """Iterate over a discrete environment."""
    for i in itertools.product(*map(range, space.nvec)):
        yield i


def combine_boxes(*spaces: Box) -> Box:
    """Combine a list of gym.Box spaces into one.

    It merges a list of unidimensional boxes to one unidimensional box by
    combining along the only dimension. Limits are kept separate.
    Output type is np.float32.
    """
    # Unidimensional spaces
    assert all(len(space.shape) == 1 for space in spaces)

    # Concat
    lows = np.concatenate([space.low for space in spaces])
    highs = np.concatenate([space.high for space in spaces])

    return Box(lows, highs)


class RewardShaper:
    r"""A Reward shaper computes a modified reward.

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
        zero_terminal_state: bool = True,
    ):
        """
        Initialize the reward shaping wrapper.

        :param value_function: the value function.
        :param mapping_function: the mapping function.
        :param gamma: MDP discount factor.
        :param zero_terminal_state: if the terminal state of
          a trajectory should have potential equal to zero.
          For details, please see:
            http://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf

        """
        self.value_function = value_function
        self.mapping_function = mapping_function
        self.gamma = gamma
        self.zero_terminal_state = zero_terminal_state

        self._last_state: StateL = None

    def reset(self, state: StateL):
        """Reset the environment.

        :param state: state returned by env.reset().
        """
        self._last_state = state

    def step(self, state_p: StateL, done: bool) -> float:
        """Do a step, and get shaping reward.

        :param state_p: new state of last transition on the environment.
        :param done: is the new state terminal?
        :return: shaping reward, it should be added to the original reward.
        """
        previous_state = self.mapping_function(self._last_state)
        current_state = self.mapping_function(state_p)
        self._last_state = state_p

        v1 = self.value_function(previous_state)
        v2 = self.value_function(current_state)
        if done and self.zero_terminal_state:
            # see http://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf
            v2 = 0
        shaping_reward = self.gamma * v2 - v1

        return shaping_reward
