"""Modular DQN policy."""
from typing import Tuple

import gym
from gym.spaces import Tuple as GymTuple, Box
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from gym.spaces import Box, Discrete
from stable_baselines.deepq.policies import FeedForwardPolicy


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
                    tf.reduce_sum(tf.matmul(automaton_onehot, action_scores), axis=1),
                    shape=(-1, self.n_actions),
                )

            q_out = final_action_scores

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
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

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


# TODO: roberto-my try. Still to run and test in training
class LnMlpModularPolicy(FeedForwardPolicy):
    """Define a Mlp model with modular states.

    This model is used when the input observation is composed by the
    environment component and a scalar automaton state. These two need to be
    treated differently. This class is adapted from
    stable_baselines.deepq.policies.FeedForwardPolicy.
    Use wrappers.temprl.JoinObservationWrapper with this policy.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
        layers, reuse=False, dueling=True):
        """Initialize the model."""

        # Base class
        super(LnMlpModularPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, dueling=dueling,
            reuse=reuse, scale=False)

        #
        act_fun = tf.nn.relu
        n_states = ob_space.high[-1]

        # Model definition
        with tf.variable_scope("model", reuse=reuse):

            # Split environment observation and state
            obs = self.processed_obs[:-1]
            state = self.processed_obs[-1]

            with tf.variable_scope("action_value"):
                action_out = obs
                # Common part
                for layer_size in layers:
                    action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                    action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                    action_out = act_fun(action_out)
                # Split by state
                action_scores = tf_layers.fully_connected(action_out, num_outputs=n_states * self.n_actions, activation_fn=None)
                action_scores = tf.reshape(action_scores, shape=(n_states, self.n_actions))
                indices = tf.reshape(state, shape=(1,))
                action_scores = tf.gather_nd(params=action_scores, indices=indices, batch_dims=1)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = obs
                    # Common part
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    # Split by state
                    state_score = tf_layers.fully_connected(state_out, num_outputs=n_states * self.n_actions, activation_fn=None)
                    state_score = tf.reshape(state_score, shape=(n_states, self.n_actions))
                    indices = tf.reshape(state, shape=(1,))
                    state_score = tf.gather_nd(params=state_score, indices=indices, batch_dims=1)
                # Dueling part
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                # Simple q
                q_out = action_scores

        self.q_values = q_out
        self._setup_init()
