"""Modular DQN policy."""
from typing import Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from gym.spaces import Box, Discrete
from gym.spaces import Tuple as GymTuple
from stable_baselines.deepq import DQN
from stable_baselines.deepq.policies import DQNPolicy

# Module random number generator
_rng = np.random.default_rng()


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


class ModularPolicy(DQNPolicy):
    """A model similar to DQN but with separate subnets.

    This model assumes that the last entry of the observation is an index
    that selects one of N DQN subnetworks.
    """

    def __init__(
        self,
        sess,
        ob_space: Box,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        shared_layers=0,
        layers=None,
        reuse=False,
        dueling=False,
        layer_norm=False,
        act_fun=tf.nn.relu,
        obs_phs=None,
        action_bias: Optional[DQN] = None,
        action_bias_eps: float = 0.0,
    ):
        """Initialize.

        See DQNPolicy for most parameters.
        :param shared_layers: this number of layers is shared between all
            subnets.
        :param action_bias: this should be an agent with same dimensions as
            this one. It's choices will be used as a bias to select actions
            while this agent is exploring.
        :param action_bias_eps: the probability of an un-biased action.
        """
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
        # Checks
        assert not dueling, "Dueling not supported."
        assert 0 <= shared_layers <= len(layers)

        # Store
        self._biased_agent = action_bias
        self._biased_agent_eps = action_bias_eps

        # NOTE assumption: only one automaton component, and at the end.
        _agent_space, automaton_space = split_agent_and_automata(ob_space)

        # Default sizes
        if layers is None:
            layers = [64, 64]

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
                for i, layer_size in enumerate(layers):

                    # Shared weights
                    if i < shared_layers:
                        x = tf_layers.fully_connected(
                            x,
                            num_outputs=layer_size,
                            activation_fn=None,
                        )
                    # Partitioned layers
                    else:
                        x = fully_connected_blocks(
                            x,
                            units=layer_size,
                            blocks=automaton_space.n,
                            name="dense_blocks" + str(i),
                        )

                    # Activation
                    if layer_norm:
                        x = tf_layers.layer_norm(x, center=False, scale=False)
                    x = act_fun(x)

                # Output layer
                x = fully_connected_blocks(
                    x,
                    units=self.n_actions,
                    blocks=automaton_space.n,
                    name="dense_blocks_output",
                )

                # Select q-values based on subtask
                x = tf.transpose(x, (0, 2, 1))
                x = tf.gather_nd(x, automaton_feature, batch_dims=1)

        self.q_values = x
        self._setup_init()

    def step(self, obs, state=None, _mask=None, deterministic=True):
        # TODO: I'm still training the first agent.. Double check biased
        #   actions afterwards

        assert _mask is None
        assert state is None
        batch_size = obs.shape[0]

        # Compute deterministic action
        q_values, _actions_proba = self.sess.run(
            [self.q_values, self.policy_proba], {self.obs_ph: obs}
        )

        # Deterministic action
        if deterministic:
            actions = np.argmax(q_values, axis=1)

        # Exploration
        else:
            # Uniform probability
            #   NOTE: before it was categorical based on current Q function
            actions = _rng.integers(self.n_actions, size=batch_size)

            # Biased actions
            if self._biased_agent is not None:
                biased_actions = self._biased_agent.policy.step(obs, deterministic=True)

                bias_samples = _rng.random(size=batch_size)
                biased_choices = bias_samples >= self._biased_agent_eps

                actions = np.where(biased_choices, biased_actions, actions)

        return actions, q_values, None

    def proba_step(self, obs, state=None, _mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


def fully_connected_blocks(
    x,
    *,
    units: int,
    blocks: int,
    name: str,
    kernel_initializer=None,
    bias_initializer=None,
):
    """Define a partitioned dense layer.

    This is similar to a Dense layer with `units` outputs, but it defines
    a `blocks` number of parallel computations.
    The computation of each block is independent.
    No activation, regularization, constraints.

    :param x: input tensor of shape (batch, features)
        or (batch, features, blocks). If the input is a 2D tensor, the
        same input is used to compute the output for all blocks.
    :param units: size of each output
    :param blocks: number of outputs
    :param name: layer unique name_scope.
    :param kernel_initializer: see tf doc.
    :param bias_initializer: see tf doc.
    :return: a tensor of shape (batch_size, features, blocks)
    """
    # Set defaults
    if kernel_initializer is None:
        kernel_initializer = tf.initializers.glorot_uniform
    if bias_initializer is None:
        bias_initializer = tf.initializers.zeros

    # Name scope
    with tf.variable_scope(name):

        # Duplicate if not partitioned
        assert 2 <= x.shape.rank <= 3
        if x.shape.rank == 2:
            x = tf.tile(tf.expand_dims(x, -1), [1, 1, blocks])
        assert x.shape[2] == blocks
        features = x.shape[1]

        # Create parameters
        kernel = tf.get_variable(
            name="block_kernel",
            shape=(features, units, blocks),
            dtype=tf.float32,
            initializer=tf.initializers.glorot_uniform,
            trainable=True,
        )
        bias = tf.get_variable(
            name="block_bias",
            shape=(units, blocks),
            dtype=tf.float32,
            initializer=tf.initializers.zeros,
            trainable=True,
        )

        # Computation
        #   (b: batch, f: features, u: units, k: block)
        x = tf.einsum("bfk,fuk->buk", x, kernel)
        x = x + tf.expand_dims(bias, 0)

    return x
