# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#
"""Reward shaping wrapper."""
from typing import Any, Mapping, Optional, Tuple

from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import actions, configurations
from gym_sapientino.core.types import Colors
from temprl.types import Interpretation

from multinav.algorithms.agents import QFunctionModel
from multinav.envs.env_abstract_sapientino import OfficeAbstractSapientino
from multinav.envs.env_abstract_sapientino import OfficeFluents as AbstractOfficeFluents
from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.callbacks import Callback
from multinav.helpers.reward_shaping import StateH, StateL, ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.utils import CallbackWrapper, SingleAgentWrapper

# Global def
color2int = {str(c): i for i, c in enumerate(list(Colors))}
int2color = {i: c for c, i in color2int.items()}


class Mapping2abs:
    """Callable that transforms a state to an OfficeAbstractSapientino state."""

    def __init__(self, env: OfficeAbstractSapientino):
        """Initialize."""
        self._env = env

    def __call__(self, state: Mapping[str, Any]) -> StateH:
        """Map function from this env to abstract env."""
        color_str = int2color[state["color"]]
        if color_str in ("blank", "wall"):
            return self._env.name2location["corridor"]
        else:
            assert 0 <= state["color"] - 2 <= self._env.n_locations - 1, (
                f"error on color '{color_str}': "
                f"0 <= {state['color'] - 2} <= {self._env.n_locations - 1}")
            return state["color"] - 2


class OfficeFluents(FluentExtractor):
    """Propositions evaluation for the fluents in the grid environment.

    This only works if combined with the Callback saved in self.callback
    """

    def __init__(self, n_rooms: int, seed: int):
        """Initialize."""
        self.abstract_env = OfficeAbstractSapientino(
            n_rooms=n_rooms, p_failure=0.0, seed=seed)
        self.abstract_fluents = AbstractOfficeFluents(self.abstract_env)

        # Colors to positions
        self._map = Mapping2abs(self.abstract_env)

        # Simulate abstract sapientino at the same time
        self.callback = self.FluentsCallback(self)

        # Last valuation (support NOP)
        self._last: Interpretation = set()

    @property
    def all(self):
        """All fluents."""
        return self.abstract_fluents.all

    def __call__(self, obs: Mapping[str, Any], action: Optional[int]) -> Interpretation:
        """Compute an interpretation for the propositios.

        :param obs: dict space observation of a single sapientino robot.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        # Map to abstract
        location, abs_action = self._state_action_to_abs(obs, action)

        # Compute features with abstract
        if abs_action is not None:
            self._last = self.abstract_fluents(location, abs_action)
        return self._last

    def _state_action_to_abs(self, obs, action: Optional[int]) -> Tuple[int, Optional[int]]:
        """Map state and action to abstract environment.

        Only used as a way to compute the fluents; we could simulate fluents in
        this env, instead.

        :param obs: destination state in grid sapientino
        :param action: last action of grid sapientino
        :return tuple of action and location in abstract sapientino (in this order)
        """
        location = self._map(obs)
        if action is None:
            return location, None
        if actions.GridCommand(action) == actions.GridCommand.NOP:
            abs_action = None
        elif actions.GridCommand(action) == actions.GridCommand.BEEP:
            abs_action = self.abstract_env.action_interact
        else:
            abs_action = self.abstract_env.action_goto_state(location)
        return location, abs_action

    class FluentsCallback(Callback):
        """Simulate with abstract sapientino in parallel."""

        def __init__(self, features_extractor: "OfficeFluents"):
            """Initialize."""
            self._extractor = features_extractor

        def _on_reset(self, _obs: StateL) -> None:
            """Reset."""
            self._extractor.abstract_env.reset()

        def _on_step(self, action, obs, _reward, _done, _info) -> None:
            """Step."""
            _, abs_action = self._extractor._state_action_to_abs(obs, action)
            if abs_action is not None:
                self._extractor.abstract_env.step(abs_action)


def abs_sapientino_shaper(path: str, n_rooms: int, seed: int) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `OfficeAbstractSapientino` then it uses it to
    compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint.
    :param n_rooms: number of rooms in the env where the agent was trained.
    :return: reward shaper to apply.
    """
    # Trained agent on abstract environment
    agent = QFunctionModel.load(path=path)

    # Mapping
    mapping = Mapping2abs(OfficeAbstractSapientino(n_rooms=n_rooms, p_failure=0.0, seed=seed))

    def map_with_temporal_goals(state: Tuple[Mapping[str, Any], list]) -> StateH:
        obs = mapping(state[0])
        qs = state[1]
        return obs, *qs

    # Shaper
    shaper = ValueFunctionRS(
        value_function=lambda s: agent.q_function[s].max(),
        mapping_function=map_with_temporal_goals,
        gamma=1.0,  # this is intentional
        zero_terminal_state=False,  # this is intentional
    )

    return shaper


def make(params: Mapping[str, Any], log_dir: Optional[str] = None):
    """Make the sapientino grid state environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: a gym Environemnt.
    """
    # Define the robot
    agent_configuration = configurations.SapientinoAgentConfiguration(
        initial_position=params["initial_position"],
        commands=actions.GridCommand,
    )

    # Define the environment
    configuration = configurations.SapientinoConfiguration(
        (agent_configuration,),
        grid_map=params["map"],
        reward_outside_grid=params["reward_outside_grid"],
        reward_duplicate_beep=params["reward_duplicate_beep"],
        reward_per_step=params["reward_per_step"],
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    fluent_extractor = OfficeFluents(
        n_rooms=params["nb_rooms"], seed=params["seed"])

    # Update fluents
    env = CallbackWrapper(env, fluent_extractor.callback)

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
        must_load=True,
    )

    # Time limit (this should be before reward shaping)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    # Reward shaping on previous envs
    if params["shaping"]:
        abs_shaper = abs_sapientino_shaper(
            path=params["shaping"],
            n_rooms=params["nb_rooms"],
            seed=params["seed"],
        )
        env = RewardShapingWrapper(env, reward_shaper=abs_shaper)

    # Choose the environment features
    env = GridRobotFeatures(env)

    return env
