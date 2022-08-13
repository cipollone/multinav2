"""Continuous control on rooms environment."""

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from gym.envs.registration import EnvSpec
from gym.wrappers import TimeLimit
from gym import Env
from gym_sapientino import SapientinoDictSpace
from gym_sapientino.core import actions, configurations

from multinav import starting_cwd
from multinav.algorithms.agents import QFunctionModel
from multinav.envs.env_grid_rooms import GridOfficeFluents as ContOfficeFluents
from multinav.envs.env_grid_rooms import GridRoomsFluents as ContRoomsFluents
from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.reward_shaping import ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper, RewardShift
from multinav.wrappers.sapientino import ContinuousRobotFeatures, GridRobotFeatures
from multinav.wrappers.utils import FailProbability, SingleAgentWrapper

logger = logging.getLogger(__name__)


def grid_rooms_shaper(
    env: Env,
    path: str,
    gamma: float,
    return_invariant: bool,
) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `GridRooms` then it uses it to
    compute the reward shaping to apply to this environment.

    :param env: sapientino dict space environment.
    :param path: path to saved checkpoint.
    :param gamma: discount factor to apply for shaping.
    :param return_invariant: if true, we apply classic return-invariant reward shaping.
        We usually want this to be false.
    :return: reward shaper to apply.
    """
    # Trained agent on abstract environment
    full_path = starting_cwd / Path(path)   # Rllib modifies path: restore
    agent = QFunctionModel.load(path=str(full_path))

    # Compute value according to grid agent
    grid_env = GridRobotFeatures(env)

    def grid_value(s) -> float:
        grid_state = grid_env._process_state(s)
        return agent.q_function[grid_state].max()

    # Shaper
    shaper = ValueFunctionRS(
        value_function=grid_value,
        mapping_function=lambda s: s,
        gamma=gamma,
        zero_terminal_state=return_invariant,
    )

    return shaper


def make(params: Mapping[str, Any], log_dir: Optional[str] = None):
    """Make the grid_rooms environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: a gym Environemnt.
    """
    # Define the robot
    agent_configuration = configurations.SapientinoAgentConfiguration(
        initial_position=params["initial_position"],
        commands=actions.ContinuousCommand,
        angular_speed=params["angular_speed"],
        acceleration=params["acceleration"],
        max_velocity=params["max_velocity"],
        min_velocity=params["min_velocity"],
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
    env.unwrapped.spec = EnvSpec("SapientinoCont-v0")  # Later wrappers may write in spec

    # Fail probability
    if params["fail_p"] > 0:
        env = FailProbability(env, fail_p=params["fail_p"], seed=params["seed"])

    # Fluents for this environment
    fluent_extractor: FluentExtractor
    if params["fluents"] == "rooms":
        fluent_extractor = ContRoomsFluents(map_config=params["map"])
    elif params["fluents"] == "party":
        raise NotImplementedError
    elif params["fluents"] == "office":
        fluent_extractor = ContOfficeFluents(
            rooms_connectivity=params["rooms_connectivity"],
            rooms_and_colors=params["rooms_and_colors"],
            interact_action=int(actions.ContinuousCommand.BEEP.value),
            seed=params["seed"],
        )
    else:
        raise ValueError(params["fluents"])

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
        must_load=True,
    )

    # Reward shift
    if params["reward_shift"] != 0:
        env = RewardShift(env, params["reward_shift"])

    # Time limit (this should be before reward shaping, also a spec is mandatory)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])
    assert env.spec.max_episode_steps == params["episode_time_limit"]

    # Reward shaping on previous envs
    if params["shaping"]:
        grid_shaper = grid_rooms_shaper(
            env=env,
            path=params["shaping"],
            gamma=params["shaping_gamma"],
            return_invariant=params["return_invariant"],
        )
        env = RewardShapingWrapper(env, reward_shaper=grid_shaper)

    # Choose the environment features
    env = ContinuousRobotFeatures(env)

    return env
