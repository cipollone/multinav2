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
        _agent_space, automaton_space = split_agent_and_automata(ob_space)

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            # extracted_features = tf.layers.flatten(self.processed_obs)
            # nb_features = int(extracted_features.shape[1])
            # agent_features, automaton_feature = tf.split(
            #     extracted_features, num_or_size_splits=[nb_features - 1, 1], axis=1
            # )
            # automaton_feature = tf.cast(automaton_feature, tf.int64)
            agent_features, automaton_feature = (
                self.processed_obs[:, :-1],
                self.processed_obs[:, -1],
            )
            automaton_feature = tf.reshape(
                tf.cast(automaton_feature, tf.int64), shape=(-1, 1)
            )
            final_q_action_out = []
            with tf.variable_scope("action_value"):
                for _q in range(automaton_space.n):
                    q_action_out = agent_features
                    for layer_size in layers:
                        q_action_out = tf_layers.fully_connected(
                            q_action_out,
                            num_outputs=layer_size,
                            activation_fn=None,
                        )
                        if layer_norm:
                            q_action_out = tf_layers.layer_norm(
                                q_action_out, center=True, scale=True
                            )
                        q_action_out = act_fun(q_action_out)

                    q_action_out = tf_layers.fully_connected(
                        q_action_out,
                        num_outputs=self.n_actions,
                        activation_fn=None,
                    )
                    final_q_action_out.append(q_action_out)

                indices = tf.reshape(automaton_feature, shape=(-1, 1, 1))
                action_scores = tf.stack(final_q_action_out, axis=1)
                final_action_scores = tf.gather_nd(
                    params=action_scores, indices=indices, batch_dims=1
                )
                final_action_scores = tf.reshape(
                    final_action_scores, shape=(-1, self.n_actions)
                )

            q_out = final_action_scores

        self.q_values = q_out
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
