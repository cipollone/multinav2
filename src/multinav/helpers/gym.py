"""This module contains helpers related to OpenAI Gym."""
import itertools
import random
import shutil
import time
from copy import deepcopy
from functools import singledispatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import gym
import numpy as np
from graphviz import Digraph
from gym import Wrapper
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.spaces import Discrete, MultiDiscrete
from gym.wrappers import TimeLimit
from PIL import Image

State = Any
Action = int
Probability = float
Reward = float
Done = bool
Transition = Tuple[Probability, State, Reward, Done]
Transitions = Dict[State, Dict[Action, List[Transition]]]


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
        assert 0 <= state < self.nS, f"{state} is not a legal state."

    def _is_legal_action(self, action: Action):
        """Check that it is a legal action."""
        assert 0 <= action < self.nA, f"{action} is not a legal action."

    def available_actions(self, state):
        """Get the available action from a state."""
        self._is_legal_state(state)
        actions = set()
        for action, _transitions in self.P.get(state, {}).items():
            actions.add(action)
        return actions


class MyMonitor(Wrapper):
    """A simple monitor."""

    def __init__(self, env: gym.Env, directory: str, force: bool = False):
        """
        Initialize the environment.

        :param env: the environment.
        :param directory: the directory where to save elements.
        """
        super().__init__(env)

        self._directory = Path(directory)
        shutil.rmtree(directory, ignore_errors=force)
        self._directory.mkdir(exist_ok=False)

        self._current_step = 0
        self._current_episode = 0

    def _save_image(self):
        """Save a frame."""
        array = self.render(mode="rgb_array")
        image = Image.fromarray(array)
        episode_dir = f"{self._current_episode:05d}"
        filepath = f"{self._current_step:05d}.jpeg"
        (self._directory / episode_dir).mkdir(parents=True, exist_ok=True)
        image.save(str(self._directory / episode_dir / filepath))

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset(**kwargs)
        self._current_step = 0
        self._current_episode += 1
        self._save_image()
        return result

    def step(self, action):
        """Do a step in the environment, and record the frame."""
        result = super().step(action)
        self._current_step += 1
        self._save_image()
        return result


class MyStatsRecorder(gym.Wrapper):
    """Stats recorder."""

    def __init__(self, env: gym.Env):
        """Initialize stats recorder."""
        super().__init__(env)
        self.episode_lengths: List[int] = []
        self.episode_rewards: List[float] = []
        self.timestamps: List[int] = []
        self.steps = None
        self.total_steps = 0
        self.rewards = None

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        self.steps += 1
        self.total_steps += 1
        self.rewards += reward
        self.done = done
        if done:
            self.save_complete()

        return state, reward, done, info

    def save_complete(self):
        """Save episode statistics."""
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.rewards))
            self.timestamps.append(time.time())

    def reset(self, **kwargs):
        """Do reset."""
        result = super().reset(**kwargs)
        self.done = False
        self.steps = 0
        self.rewards = 0
        return result


def _random_action(env: gym.Env, state):
    available_actions = getattr(
        env, "available_actions", lambda _: list(iter_space(env.action_space))
    )
    actions = available_actions(state)
    return random.choice(list(actions))


def rollout(
    env: gym.Env,
    nb_episodes: int = 1,
    max_steps: int = 10,
    policy=lambda env, state: _random_action,
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
