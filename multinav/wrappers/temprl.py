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
"""Helpers related to TempRL wrappers."""

import numpy as np
from gym import Env, ObservationWrapper
from gym.spaces import Discrete, MultiDiscrete, Tuple as GymTuple


class FlattenAutomataStates(ObservationWrapper):
    """Flatten the observation space to one array.

    A state (x, [q1, q2]) becomes (x, q1, q2).
    Discrete features x.
    """

    def __init__(self, env: Env):
        """Initialize.

        :param env: gym environment to wrap.
        """
        ObservationWrapper.__init__(self, env)

        space = env.observation_space
        assert isinstance(space, GymTuple), "Unexpectd gym space"
        assert len(space) == 2, "Expected: environment state, automata states"
        assert type(space[0]) == Discrete, "Env state must be discrete"
        assert type(space[1]) == MultiDiscrete, "Automata states are discrete"

        self.observation_space = MultiDiscrete(
            (np.insert(space[1].nvec, 0, space[0].n))
        )

    def observation(self, observation):
        """Flatten."""
        env_state = observation[0]
        automata_states = tuple(observation[1])
        return (env_state,) + automata_states
