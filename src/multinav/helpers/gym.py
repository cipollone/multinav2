"""This module contains helpers related to OpenAI Gym."""
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gym
import numpy as np
from graphviz import Digraph
from gym import Wrapper
from gym.envs.toy_text.discrete import DiscreteEnv
from PIL import Image

State = int
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
