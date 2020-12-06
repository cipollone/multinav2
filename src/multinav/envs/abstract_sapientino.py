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

"""This package contains the implementation of an 'abstract' Sapientino with teleport."""
import io

import numpy as np
from PIL import Image

from src.multinav.helpers.gym import from_discrete_env_to_graphviz, MyDiscreteEnv, Action, Transitions, State


class AbstractSapientino(MyDiscreteEnv):
    """Abstract Sapientino environment."""

    def __init__(self, nb_colors: int):
        """
        Initialize the environment.

        :param nb_colors: the number of color to consider.
        """
        self._nb_colors = nb_colors
        model = self._make_transitions()
        ids = np.zeros(self.nb_states)
        ids[self.initial_state] = 1.0
        super().__init__(self.nb_states, self.nb_actions, model, ids)

    @property
    def nb_colors(self) -> int:
        """Get the number of colors."""
        return self._nb_colors

    @property
    def initial_state(self) -> int:
        """Get the initial state."""
        return 0

    @property
    def nb_states(self) -> int:
        """
        Get the number of states.

        That is:
        - one state for each color
        - one state for the corridor

        :return: the number of states.
        """
        return self.nb_colors + 1

    @property
    def nb_actions(self) -> int:
        """
        Get the number of actions.

        It includes:
        - go to to each color (+nb_colors)
        - go back to the corridor (+1)
        - visit color (+1)

        :return: the number of actions.
        """
        return self.nb_colors + 1 + 1

    def _is_legal_color(self, color_id: int):
        """Check that it is a legal color."""
        assert 0 <= color_id < self.nb_colors, f"{color_id} is not a legal color."

    def state_from_color(self, color_id: int):
        """
        Get the state from the color.

        Sum +1 because 0 is the initial state.
        """
        self._is_legal_color(color_id)
        return color_id + 1

    def action_goto_color_from_color(self, color_id: int) -> Action:
        """Get the action "goto color" from the color id."""
        assert 0 <= color_id < self.nb_colors, f"{color_id} is not a legal color."
        return color_id + 2

    @property
    def goto_corridor(self) -> Action:
        """Get the action "goto corridor"."""
        return 0

    @property
    def visit_color(self) -> Action:
        """Get the action "visit_color"."""
        return 1

    def _make_transitions(self) -> Transitions:
        """
        Make the model.

        :return: the transitions.
        """
        model: Transitions = {}

        for color_id in range(self.nb_colors):
            color_state = self.state_from_color(color_id)

            # from the corridor, you can go to any color
            goto_color_action = self.action_goto_color_from_color(color_id)
            new_transition = (1.0, color_state, 0.0, False)
            model.setdefault(self.initial_state, {}).setdefault(
                goto_color_action, []
            ).append(new_transition)

            # if you visit a color, you remain in the same state.
            new_transition = (1.0, color_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(self.visit_color, []).append(
                new_transition
            )

            # from any color, you can go back to the corridor.
            new_transition = (1.0, self.initial_state, 0.0, False)
            model.setdefault(color_state, {}).setdefault(self.goto_corridor, []).append(
                new_transition
            )

        return model

    def _action_to_string(self, action: Action):
        """From action to string."""
        self._is_legal_action(action)
        if action == self.goto_corridor:
            return "goto_corridor"
        if action == self.visit_color:
            return "visit"
        return f"goto_{action}"

    def _state_to_string(self, state: State):
        """From state to string.""" ""
        # TODO add color strings.
        self._is_legal_state(state)
        return str(state)

    def render(self, mode="human"):
        assert mode == "rgb_array"
        graph = from_discrete_env_to_graphviz(
            self, state2str=self._state_to_string, action2str=self._action_to_string
        )
        image_file = io.BytesIO(graph.pipe(format="png"))
        array = np.array(Image.open(image_file))
        return array
