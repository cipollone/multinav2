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
"""Helper functions for the notebooks."""
import io
from io import BytesIO

import gym
import matplotlib.pyplot as plt
import numpy as np
import PIL
from IPython.core.display import Image, display
from pythomata.dfa import DFA

from multinav.helpers.pythomata import to_graphviz_sym


def display_img_array(ima):
    """Display an rgb_array."""
    im = PIL.Image.fromarray(ima)
    bio = BytesIO()
    im.save(bio, format="png")
    display(Image(bio.getvalue(), format="png"))


def print_env(env: gym.Env, *_args):
    """Print the OpenAI Gym environment in Jupyter."""
    display_img_array(env.render(mode="rgb_array"))


def plot_env(env: gym.Env, *_args, **_kwargs):
    """Print the OpenAI Gym environment in Jupyter."""
    plt.figure()
    plt.imshow(env.render(mode="rgb_array"))
    plt.show()


def automaton_to_rgb(dfa: DFA, **kwargs):
    """Automaton to RGB array."""
    graph = to_graphviz_sym(dfa, **kwargs)
    image_file = io.BytesIO(graph.pipe(format="png"))
    array = np.array(PIL.Image.open(image_file))
    return array


def print_automaton(dfa: DFA):
    """Print the automaton in Jupyter."""
    array = automaton_to_rgb(dfa)
    display_img_array(array)
