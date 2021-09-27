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
from typing import Any, Mapping, Optional

from gym.wrappers import TimeLimit
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import actions, configurations
from gym_sapientino.core.types import Colors
from temprl.types import Interpretation

from multinav.algorithms.agents import ValueFunctionModel
from multinav.envs import env_abstract_sapientino, temporal_goals
from multinav.helpers.callbacks import Callback
from multinav.helpers.reward_shaping import StateH, StateL, ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper
from multinav.wrappers.sapientino import GridRobotFeatures
from multinav.wrappers.utils import CallbackWrapper, SingleAgentWrapper

# Global def
color2int = {str(c): i for i, c in enumerate(list(Colors))}
int2color = {i: c for c, i in color2int.items()}


# TODO
class GridSapientinoRewardShaper(ValueFunctionRS):
    """Reward shaper for grid Sapientino."""

    def __value_function_callable(self, state):
        return self.value_function_table[state]

    def __mapping_function(self, state):
        agent_state, automata_states = state[0], state[1:]
        color = agent_state["color"]
        return (color,) + tuple(*automata_states)

    def __init__(self, value_function_table, gamma):
        """Initialize the Sapientino reward shaper."""
        self.value_function_table = value_function_table
        ValueFunctionRS.__init__(
            self,
            value_function=self.__value_function_callable,
            mapping_function=self.__mapping_function,
            gamma=gamma,
            zero_terminal_state=False,
        )


class Mapping2abs:
    """Callable that transforms a state to an OfficeAbstractSapientino state."""

    def __init__(self, env: env_abstract_sapientino.OfficeAbstractSapientino):
        """Initialize."""
        self._env = env

    def __call__(self, state: Mapping[str, Any]) -> StateH:
        """Map function from this env to abstract env."""
        color_str = int2color[state["color"]]
        if color_str in ("blank", "wall"):
            return self._env.name2location["corridor"]
        else:
            assert 0 <= state["color"] - 2 <= self._env.n_locations - 1, (
                f"0 <= {state['color'] - 2} <= {self._env.n_locations - 1}")
            return state["color"] - 2


class OfficeFluents(temporal_goals.FluentExtractor):
    """Propositions evaluation for the fluents in the grid environment.

    This only works if combined with the Callback saved in self.callback
    """

    def __init__(self, n_rooms: int, seed: int):
        """Initialize."""
        self.abstract_env = env_abstract_sapientino.OfficeAbstractSapientino(
            n_rooms=n_rooms, p_failure=0.0, seed=seed)
        self.abstract_fluents = env_abstract_sapientino.OfficeFluents(
            self.abstract_env)

        # Colors to positions
        self._map = Mapping2abs(self.abstract_env)

        # Simulate abstract sapientino at the same time
        self.callback = self.FluentsCallback(self)

    @property
    def all(self):
        """All fluents."""
        return self.abstract_fluents.all

    def __call__(self, obs: Mapping[str, Any], action: int) -> Interpretation:
        """Compute an interpretation for the propositios.

        :param obs: dict space observation of a single sapientino robot.
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        # Get location in abstract
        location = self._map(obs)

        # Compute features with abstract
        print("features: ", self.abstract_fluents(location, action))
        return self.abstract_fluents(location, action)

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
            abs_env = self._extractor.abstract_env
            if actions.GridCommand(action) == actions.GridCommand.NOP:
                return
            if actions.GridCommand(action) == actions.GridCommand.BEEP:
                abs_action = abs_env.action_interact
            else:
                location = self._extractor._map(obs)
                abs_action = self._extractor.abstract_env.action_goto_state(location)
            abs_env.step(abs_action)
            # TODO: debug


# TODO
def abs_sapientino_shaper(path: str, gamma: float) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `AbstractSapientinoTemporalGoal` then
    it uses it to compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint for `AbstractSapientinoTemporalGoal`
    :param gamma: RL discount factor.
    :return: reward shaper to apply.
    """
    # AbstractSapientinoTemporalGoal's agent is a ValueFunctionModel
    agent = ValueFunctionModel.load(path=path)

    # Define mapping
    def _map(state: StateL) -> StateH:
        agent_state, automata_states = state[0], state[1:]
        color = agent_state["color"]
        # Wall (color == 1) is not mapped to anything here
        if color != 0:
            color -= 1
        return (color,) + tuple(*automata_states)

    # Shaper
    shaper = ValueFunctionRS(
        value_function=lambda s: agent.value_function[s],
        mapping_function=_map,
        gamma=1.0,   # NOTE: this is intentional
        zero_terminal_state=False,  # NOTE ^
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
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        reward_per_step=0.0,
    )
    env = SingleAgentWrapper(SapientinoDictSpace(configuration))

    # Define the fluent extractor
    fluent_extractor = OfficeFluents(
        n_rooms=params["nb_rooms"], seed=params["seed"])

    # Update fluents
    env = CallbackWrapper(env, fluent_extractor.callback)

    # Apply temporal goals to this env
    env = temporal_goals.with_nonmarkov_rewards(
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
            gamma=params["gamma"],
        )
        env = RewardShapingWrapper(env, reward_shaper=abs_shaper)

    # Choose the environment features
    env = GridRobotFeatures(env)

    return env
