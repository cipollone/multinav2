"""Modular DQN policy."""
from typing import Tuple

import gym
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from gym.spaces import Box, Discrete
from stable_baselines.deepq.policies import DQNPolicy


def split_agent_and_automata(ob_space: Box) -> Tuple[gym.Space, Discrete]:
    """
    Split agent and the automaton space.

    TODO: for the moment, we assume the automata spaces are composed by
      just one component.

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
        # TODO assumption: only one automaton component, and at the end.
        agent_space, automaton_space = split_agent_and_automata(ob_space)

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                extracted_features = tf.layers.flatten(self.processed_obs)
                nb_features = int(extracted_features.shape[1])
                agent_features, automaton_feature = tf.split(
                    extracted_features, num_or_size_splits=[nb_features - 1, 1], axis=1
                )
                automaton_feature = tf.cast(automaton_feature, tf.int64)
                automaton_onehot = tf.reshape(
                    tf.one_hot([automaton_feature], depth=automaton_space.n),
                    shape=(-1, automaton_space.n),
                )

                q_action_out = tf.tile(
                    agent_features[:, None, :], (1, automaton_space.n, 1)
                )
                for layer_size in layers:
                    q_action_out = tf_layers.fully_connected(
                        q_action_out, num_outputs=layer_size, activation_fn=None
                    )
                    if layer_norm:
                        q_action_out = tf_layers.layer_norm(
                            q_action_out, center=True, scale=True
                        )
                    q_action_out = act_fun(q_action_out)

                action_scores = tf_layers.fully_connected(
                    q_action_out, num_outputs=self.n_actions, activation_fn=None
                )
                final_action_scores = tf.reshape(
                    tf.matmul(automaton_onehot, action_scores),
                    shape=(-1, automaton_space.n),
                )

            q_out = final_action_scores

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        pass

    def proba_step(self, obs, state=None, mask=None):
        pass
