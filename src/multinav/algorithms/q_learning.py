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
from typing import Any, Callable, Dict, Optional

import gym
import numpy as np

logger = logging.getLogger(__name__)


def _random_action(nb_actions):
    """Random actions."""
    return (
        np.random.randn(
            nb_actions,
        )
        * sys.float_info.epsilon
    )


def q_learning(
    env: gym.Env,
    total_timesteps: int = 2000000,
    alpha: float = 0.1,
    eps: float = 0.1,
    gamma: float = 0.9,
    learning_rate_decay: bool = False,
    epsilon_decay: bool = False,
    epsilon_end: float = 0.0,
    learning_rate_end: float = 0.0,
    update_extras: Optional[Callable] = None,
) -> Dict[Any, np.ndarray]:
    """
    Learn a Q-function from a Gym env using vanilla Q-Learning.

    :param update_extras: an optional callable that can be used to
        log values during training.
    :return the Q function: a dictionary from states to array of Q values for every action.
    """
    # Vars
    alpha0 = alpha
    eps0 = eps

    # Init
    nb_actions = env.action_space.n
    Q: Dict[Any, np.ndarray] = defaultdict(partial(_random_action, nb_actions))

    def choose_action(state):
        if np.random.random() < eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(Q[state])

    done = True
    for step in range(total_timesteps):

        if done:
            state = env.reset()
            done = False

        # Step
        action = choose_action(state)
        state2, reward, done, _info = env.step(action)

        # Compute update
        td_update = reward + gamma * np.max(Q[state2]) - Q[state][action]

        logger.debug(
            f"Q[{state}][{action}] = {Q[state][action]} "
            f"-> {reward + gamma * np.max(Q[state2])}"
        )

        # Apply
        Q[state][action] += alpha * td_update
        state = state2

        # Log
        if update_extras:
            update_extras(td=abs(td_update))

        # Decays
        if step % 10 == 0:
            frac = step / total_timesteps
            if learning_rate_decay:
                alpha = alpha0 * (1 - frac) + learning_rate_end * frac
            if epsilon_decay:
                eps = eps0 * (1 - frac) + epsilon_end * frac

            # Log
            print(" Eps:", round(eps, 3), end="\r")

    print()
    return Q
