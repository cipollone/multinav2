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
"""Interfaces of agents computed from algorithms of this model.

These classes are just interfaces of agents that have been computed from these
algorithms.
"""
import logging
import pickle
from typing import Any, Dict

import numpy as np
import ray.rllib.agents.registry
from typing_extensions import Protocol

from multinav.helpers.gym import Action, State

logger = logging.getLogger(__name__)

QTableType = Dict[State, np.ndarray]


class AgentModel(Protocol):
    """A generic agent that can act."""

    def predict(self, observation: State) -> Action:
        """Compute the action to execute."""
        raise NotImplementedError("Abstract")

    def save(self, path: str) -> str:
        """Save model to path.

        :param path: directory + model name
        :return: the output file
        """
        raise NotImplementedError("Abstract")

    def load(self, path: str) -> "AgentModel":
        """Load model from path.

        :param path: exact file path to load.
        :return: the built agent
        """
        raise NotImplementedError("Abstract")


class QFunctionModel(AgentModel):
    """An agent that act from a Q function."""

    file_ext = ".pickle"

    def __init__(self, q_function: Dict[State, np.ndarray]):
        """Initialize.

        :param q_function: any q_function.
        """
        self.q_function = q_function

    def predict(self, observation: State) -> Action:
        """Compute best action from q function."""
        logger.debug("value: {}".format(np.max(self.q_function[observation])))
        return np.argmax(self.q_function[observation]).item()

    def save(self, path: str) -> str:
        """Save model to path (appends file_ext)."""
        full_path = path + self.file_ext
        with open(full_path, "wb") as f:
            pickle.dump(self.q_function, f, protocol=4)
        return full_path

    @staticmethod
    def load(path: str) -> "QFunctionModel":
        """Load model from exact path."""
        with open(path, "rb") as f:
            q_function = pickle.load(f)
        return QFunctionModel(q_function)


class ValueFunctionModel(AgentModel):
    """Act from a value function."""

    file_ext = ".pickle"

    def __init__(
        self,
        value_function: Dict[State, float],
        policy: Dict[State, int],
    ):
        """Initialize.

        :param value_function: any value function.
        :param policy: this agent requires a policy because it would need
            the environment model to compute the best action otherwise.
        """
        self.value_function = value_function
        self.policy = policy

    def predict(self, observation: State) -> Action:
        """Just execute the policy."""
        return self.policy[observation]

    def save(self, path: str) -> str:
        """Save model to path (appends file_ext)."""
        data = dict(
            value_function=self.value_function,
            policy=self.policy,
        )
        full_path = path + self.file_ext
        with open(full_path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        return full_path

    @staticmethod
    def load(path: str) -> "ValueFunctionModel":
        """Load model from exact path."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return ValueFunctionModel(**data)


class RllibAgentModel(AgentModel):
    """Act from a trained Rllib agent."""

    def __init__(self, agent_type: str, agent_conf: Dict[str, Any]):
        """Initialize.

        See TrainRllib for an explanation of arguments.
        """
        self.agent_type = agent_type
        self.agent_conf = agent_conf
        self.trainer = None

    def load(self, path: str) -> "RllibAgentModel":
        """Restore from checkpoint."""
        conf = dict(self.agent_conf)
        conf["num_workers"] = 0  # This creates the environment on local worker
        trainer_class = ray.rllib.agents.registry.get_trainer_class(self.agent_type)
        self.trainer = trainer_class(conf)
        self.trainer.restore(path)
        return self

    def save(self, path: str) -> str:
        """Can't save this."""
        raise NotImplementedError("Just user trainer checkpoints.")

    def predict(self, observations: State) -> Action:
        """Compute next action."""
        return self.trainer.compute_single_action(observations)
