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
from typing import Dict

import numpy as np
from typing_extensions import Protocol

from multinav.helpers.gym import Action, State

logger = logging.getLogger(__name__)


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

    @classmethod
    def load(cls, path: str) -> "AgentModel":
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
            pickle.dump(self.q_function, f)
        return full_path

    @classmethod
    def load(cls, path: str) -> "QFunctionModel":
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
            pickle.dump(data, f)
        return full_path

    @classmethod
    def load(cls, path: str) -> "ValueFunctionModel":
        """Load model from exact path."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return ValueFunctionModel(**data)
