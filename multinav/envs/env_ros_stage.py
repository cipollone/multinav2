"""Environment that interacts with a running instance of ROS stage simulator."""
import logging
from typing import Any, Mapping, Optional

import numpy as np
from gym.wrappers import TimeLimit
from rosstagerl.envs import RosControlsEnv
from temprl.types import Interpretation

from multinav.algorithms.agents import QFunctionModel
from multinav.envs.temporal_goals import FluentExtractor, with_nonmarkov_rewards
from multinav.helpers.reward_shaping import ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper, RewardShift

logger = logging.getLogger(__name__)


def grid_rooms_shaper(
    path: str,
    gamma: float,
    return_invariant: bool,
) -> ValueFunctionRS:
    """Define a reward shaper on the previous environment.

    This loads a saved agent for `GridRooms` then it uses it to
    compute the reward shaping to apply to this environment.

    :param path: path to saved checkpoint.
    :param gamma: discount factor to apply for shaping.
    :param return_invariant: if true, we apply classic return-invariant reward shaping.
        We usually want this to be false.
    :return: reward shaper to apply.
    """
    # Trained agent on abstract environment
    agent = QFunctionModel.load(path=path)

    def mapping_fn(state):
        """Return discrete position in map."""
        # TODO
        return state

    # Shaper
    shaper = ValueFunctionRS(
        value_function=lambda s: agent.q_function[s].max(),
        mapping_function=mapping_fn,
        gamma=gamma,
        zero_terminal_state=return_invariant,
    )

    return shaper


class RosPartyFluents(FluentExtractor):
    """Define propositions for Cocktail Parti on Ros-Stage environment."""

    # Positions on the map
    montreal_cp_locations = {
        "alice": np.array([8.5, -0.5, 45]),
        "carol": np.array([9.30, -5, -90]),
        "box": np.array([5, -5, 180]),
    }

    def __init__(self):
        """Initialize."""
        self.fluents = {"at_" + location for location in self.montreal_cp_locations}

    @property
    def all(self):
        """All fluents."""
        return self.fluents

    def _closeto(self, state, location: str) -> bool:
        """Return whether agent is close to a location."""
        assert location in self.montreal_cp_locations
        desired_pose = self.montreal_cp_locations[location]
        # Only positions for now
        return (np.abs(state[:2] - desired_pose[:2]) < 0.1).all()

    def __call__(self, obs, action: int) -> Interpretation:
        """Respect temprl.types.FluentExtractor interface.

        :param obs: assuming that the observation is [x, y, angle, ...]
        :param action: the last action.
        :return: current propositional interpretation of fluents
        """
        fluents = set()
        for location in self.montreal_cp_locations:
            if self._closeto(state=obs, location=location):
                fluents.add("at_" + location)
        assert fluents.issubset(self.fluents)
        logger.debug(f"Fluents for observation {obs}:\n" + str(fluents))
        return fluents


def make(params: Mapping[str, Any], log_dir: Optional[str] = None):
    """Make the ros-stage environment.

    This assumes that a robot connector is running.
    Execute scripts/ros-stage/docker-start.bash and
    scripts/ros-stage/start-connector.bash.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: a gym Environemnt.
    """
    # NOTE: Assuming simulator is running:
    #  Execute scripts/

    # Connect to simulator
    env = RosControlsEnv(
        n_actions=params["n_actions"],
        n_observations=params["n_observations"],
    )

    # Fluents for this environment
    if params["fluents"] == "rooms":
        raise NotImplementedError
    elif params["fluents"] == "party":
        fluent_extractor = RosPartyFluents()
    else:
        raise ValueError(params["fluents"])

    # Apply temporal goals to this env
    env = with_nonmarkov_rewards(
        env=env,
        rewards=params["rewards"],
        fluents=fluent_extractor,
        log_dir=log_dir,
        must_load=False,  # TODO: set to true once the levels above are ready and load from automaton
    )

    # Reward shift
    if params["reward_shift"] != 0:
        env = RewardShift(env, params["reward_shift"])

    # Time limit (this should be before reward shaping)
    env = TimeLimit(env, max_episode_steps=params["episode_time_limit"])

    # Reward shaping on previous envs
    if params["shaping"]:
        grid_shaper = grid_rooms_shaper(
            path=params["shaping"],
            gamma=params["shaping_gamma"],
            return_invariant=params["return_invariant"],
        )
        env = RewardShapingWrapper(env, reward_shaper=grid_shaper)

    return env
