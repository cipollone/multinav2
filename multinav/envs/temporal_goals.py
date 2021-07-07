"""Definition of temporal goals."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from gym import Env
from logaut import ldl2dfa, ltl2dfa
from pylogics.parsers import parse_ldl, parse_ltl
from temprl.types import FluentExtractor
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper

from multinav.wrappers.temprl import FlattenAutomataStates


def with_nonmarkov_rewards(
    env: Env,
    rewards: List[Dict[str, Any]],
    fluents: FluentExtractor,
    log_dir: Optional[str],
):
    """Wrap an environment with the specified temporal goals.

    :param env: the environment to wrap.
    :param rewards: dict parameters that specify the temporal goals with
        associated rewards.
    :param fluents: a fluent extractor. Make sure that this is consistent with
        the propositions that appear in the temporal goals.
    :param log_dir: directory where to save reward machines.
    :return: a wrapped environment with observation space (obs, q0, .., qN)
        for N temporal goals.
    """
    # Compute or load automata
    for reward_spec in rewards:
        if "ldlf" in reward_spec:
            reward_spec["dfa"] = ldl2dfa(parse_ldl(reward_spec["ldl"]))
        elif "ltlf" in reward_spec:
            reward_spec["dfa"] = ltl2dfa(parse_ltl(reward_spec["ltlf"]))
        else:
            assert "dfa" in reward_spec, (
                "You must specify ldlf, ldlf, or dfa to pickled automaton")
            with open(reward_spec["dfa"], "rb") as f:
                reward_spec["dfa"] = pickle.load(f)

    temporal_goals = [
        TemporalGoal(
            automaton=reward_spec["dfa"],
            reward=reward_spec["reward"],
        ) for reward_spec in rewards
    ]

    # Move with env
    env = TemporalGoalWrapper(
        env=env,
        temp_goals=temporal_goals,
        fluent_extractor=fluents,
    )

    # Save dfa
    if log_dir is not None:
        for i, reward_spec in enumerate(rewards):
            graph = reward_spec["dfa"].to_graphviz()
            filename = f"dfa-{i}-reward-{reward_spec['reward']}.pdf"
            filepath = Path(log_dir) / filename
            with open(filepath, "wb") as f:
                f.write(graph.pipe(format="pdf", quiet=True))

    # Simplify observation space
    env = FlattenAutomataStates(env)

    return env
