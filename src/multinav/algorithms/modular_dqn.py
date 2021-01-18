"""Modular DQN policy."""
from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from gym.spaces import Box, Discrete
from gym.spaces import Tuple as GymTuple
from stable_baselines.deepq.policies import DQNPolicy


def split_agent_and_automata(ob_space: Box) -> Tuple[gym.Space, Discrete]:
    """
    Split agent and the automaton space.

    NOTE: we assume the automata spaces are composed by just one component.

    :param ob_space: the combined observation space.
    :return: the pair of spaces (agent, automata).
    """
    lows = ob_space.low
    highs = ob_space.high

    agent_lows, automaton_low = lows[:-1], lows[-1]
    agent_highs, automaton_high = highs[:-1], highs[-1]

    assert automaton_low == 0
    nb_states = int(automaton_high) + 1
    return Box(agent_lows, agent_highs), Discrete(nb_states)


# TODO: maybe add options to allow few layers in common
class ModularPolicy(DQNPolicy):
    """Similar to DQN, but with many subnetworks"""

    def __init__(
        self,
        sess,
        ob_space: Box,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        layers=None,
        reuse=False,
        dueling=False,
        layer_norm=False,
        act_fun=tf.nn.relu,
        obs_phs=None,
    ):
        super(ModularPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            dueling=dueling,
            reuse=reuse,
            scale=False,
            obs_phs=obs_phs,
        )
        assert not dueling, "Dueling not supported."
        # NOTE assumption: only one automaton component, and at the end.
        agent_space, automaton_space = split_agent_and_automata(ob_space)

        # Default sizes
        if layers is None:
            layers = [64, 64]

        # Duplicate weights for each subtask
        layers = [units * automaton_space.n for units in layers]

        # Model scope
        with tf.variable_scope("model", reuse=reuse):

            # Split observations
            agent_features, automaton_feature = (
                self.processed_obs[:, :-1],
                self.processed_obs[:, -1],
            )
            automaton_feature = tf.reshape(
                tf.cast(automaton_feature, tf.int64), shape=(-1, 1)
            )

            # Net
            with tf.variable_scope("action_value"):
                x = agent_features

                # Layers
                for layer_size in layers:
                    x = tf_layers.fully_connected(
                        x,
                        num_outputs=layer_size,
                        activation_fn=None,
                    )
                    if layer_norm:
                        x = tf_layers.layer_norm(x, center=True, scale=True)
                    x = act_fun(x)

                # Output layer
                x = tf_layers.fully_connected(
                    x,
                    num_outputs=self.n_actions * automaton_space.n,
                    activation_fn=None,
                )

                # Select q-values based on subtask
                x = tf.reshape(x, (-1, automaton_space.n, self.n_actions))
                x = tf.gather_nd(x, automaton_feature, batch_dims=1)

        self.q_values = x
        self._setup_init()

    def step(self, obs, state=None, _mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run(
            [self.q_values, self.policy_proba], {self.obs_ph: obs}
        )
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling (see original implementation)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(
                    self.n_actions, p=actions_proba[action_idx]
                )

        return actions, q_values, None

    def proba_step(self, obs, state=None, _mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
