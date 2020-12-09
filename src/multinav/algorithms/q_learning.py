"""Q-Learning implementation."""
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Dict

import gym
import numpy as np


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
    nb_episodes: int = 100,
    alpha: float = 0.1,
    eps: float = 0.1,
    gamma: float = 0.9,
    learning_rate_decay: bool = False,
    epsilon_decay: bool = False,
) -> Dict[Any, np.ndarray]:
    """
    Learn a Q-function from a Gym env using vanilla Q-Learning.

    :return the Q function: a dictionary from states to array of Q values for every action.
    """
    t = 0
    nb_actions = env.action_space.n
    Q: Dict[Any, np.ndarray] = defaultdict(partial(_random_action, nb_actions))

    def choose_action(state):
        if np.random.random() < eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(Q[state])

    for _ in range(nb_episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state)
            state2, reward, done, info = env.step(action)
            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[state2]) - Q[state][action]
            )
            state = state2
            t += 1
            if learning_rate_decay and t != 1:
                alpha *= (t - 1) / t
            if epsilon_decay and t != 1:
                eps *= (t - 1) / t
    return Q
