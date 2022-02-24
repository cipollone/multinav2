"""Environment that interacts with a running instance of ROS stage simulator."""
import subprocess
import time
from typing import Any, Mapping, Optional

from gym.wrappers import TimeLimit
from rosstagerl.envs import RosControlsEnv

from multinav.algorithms.agents import QFunctionModel
from multinav.envs.temporal_goals import with_nonmarkov_rewards
from multinav.helpers.reward_shaping import ValueFunctionRS
from multinav.wrappers.reward_shaping import RewardShapingWrapper, RewardShift

# montreal_cp map
locations = {
    "start": [5, -5, 0],
    "alice": [8.5, -0.5, 45],
    "carol": [9.30, -5, -90],
    "box": [5, -5, 180],
}


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


def make(params: Mapping[str, Any], log_dir: Optional[str] = None):
    """Make the grid_rooms environment.

    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :param log_dir: directory where logs can be saved.
    :return: a gym Environemnt.
    """
    # Start simulator
    if "start_script" in params:
        print("Initializing")
        subprocess.run(params["start_script"], params["with_gui"])
        time.sleep(5)
        print("Initialized")

    # Connect to simulator
    env = RosControlsEnv(
        n_actions=params["n_actions"],
        n_observations=params["n_observations"],
    )
    # TODO: use action sets here

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
        must_load=True,
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
