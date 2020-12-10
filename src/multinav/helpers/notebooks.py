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


def print_env(env: gym.Env):
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
