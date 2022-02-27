"""Policies for agents with composite observations: environment + automaton states."""

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import tensorflow as tf
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow import keras, summary
from tensorflow.keras import activations, layers


def init_models():
    """Register custom models."""
    ModelCatalog.register_custom_model("composite_fc", CompositeNet)


class CompositeNet(TFModelV2):
    """Policy network for agent with composite observations.

    See CompositeFullyConnected.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
    ):
        """Initialize."""
        # Super
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )

        # For simplicity, assume only one automaton state
        assert isinstance(obs_space, spaces.Tuple)
        assert len(obs_space) == 2
        assert len(obs_space[1].nvec) == 1
        self._n_states = obs_space[1].nvec[0]

        # Inputs
        x_input = layers.Input(shape=obs_space[0].shape, name="observations")
        states_input = layers.Input(shape=(1,), name="automaton_state")
        inputs = (x_input, states_input)

        # Define model
        x = CompositeFullyConnected(
            layers_spec=model_config["layers"] + [num_outputs],
            shared_layers=model_config["shared_layers"],
            n_states=self._n_states,
            activation=model_config["activation"],
            batch_norm=model_config["batch_norm"],
            activation_last=model_config["activation"],
        )(inputs)

        self.base_model = keras.Model(inputs=inputs, outputs=x)

        # Log graph NOTE: debugging only: add @tf.function to tf_forward()
        if "log_graph" in self.model_config:
            @tf.function
            def tracing_graph(inputs):
                return self.base_model(inputs)

            graph_writer = summary.create_file_writer(self.model_config["log_graph"])
            fake_inputs = (
                np.zeros((10, *x_input.shape[1:])),
                np.zeros((10, *states_input.shape[1:]))
            )
            summary.trace_on(graph=True)
            tracing_graph(fake_inputs)
            with graph_writer.as_default():
                summary.trace_export(str(type(self)), 0)

    def forward(self, input_dict, state, seq_lens):
        """Forward pass."""
        out = self.base_model(input_dict["obs"])
        return out, state

    def value_function(self):
        """Return the value function associated to the last input."""
        raise NotImplementedError


class CompositeFullyConnected(keras.Model):
    """Fully connected layers partitioned in blocks.

    This model is a fully connected network, which is then split in separate
    branches in the final layers. Each of the separate blocks is the output
    associated to an automaton state. Each end is selected based on the current
    automaton state.
    Expected input: (numpy_array, automaton_state_index)
    """

    def __init__(
        self,
        layers_spec: Sequence[int],
        shared_layers: int,
        n_states: int,
        activation: str,
        batch_norm: bool,
        activation_last: Optional[str] = None,
    ):
        """Initialize.

        :param layers_spec: numer and size of each fully connected layer,
            output included.
        :param shared_layers: this number of layers is shared between all
            subnets.
        :param n_states: number of output partitions of the final layers
            (number of automaton states).
        :param activation: name of an activation function
        :param batch_norm: whether to use batch normalization between each layer
        :param activation_last: The activation function of the output layer
        """
        # Store
        self._layers_spec = layers_spec
        self._shared_layers = shared_layers
        self._n_states = n_states
        self._batch_norm = batch_norm
        self._activation_fn = activations.get(activation)
        self._activation_fn_last = activations.get(activation_last)
        super().__init__()

        # Define layers
        layers_list = []
        for i, layer_size in enumerate(self._layers_spec[:-1]):

            # Shared weights
            if i < self._shared_layers:
                layers_list.append(
                    layers.Dense(layer_size, activation=self._activation_fn)
                )
            # Partitioned weights
            else:
                layers_list.append(
                    FullyConnectedBlock(
                        units=layer_size,
                        blocks=self._n_states,
                        activation=self._activation_fn,
                    )
                )

            # Normalization
            if self._batch_norm:
                layers_list.append(layers.BatchNormalization())

        # Output layer
        layers_list.append(
            FullyConnectedBlock(
                units=self._layers_spec[-1],
                blocks=self._n_states,
                activation=self._activation_fn_last,
            )
        )

        # Store as sequential model
        self.forward_model = keras.Sequential(layers_list)

    def call(self, inputs):
        """Forward pass."""
        x, states = inputs[0], inputs[1]
        x = self.forward_model(x)

        # Select values based on automa state
        states = tf.reshape(tf.cast(states, tf.int64), shape=(-1, 1))
        x = tf.transpose(x, (0, 2, 1))
        x = tf.gather_nd(x, states, batch_dims=1)

        return x


class FullyConnectedBlock(layers.Layer):
    """Define a partitioned dense layer.

    This is similar to a Dense layer with `units` outputs, but it defines
    a `blocks` number of parallel computations.
    The computation of each block is independent.
    """

    def __init__(
        self,
        *,
        units: int,
        blocks: int,
        activation: Optional[Callable] = None,
        kernel_initializer=None,
        bias_initializer=None,
    ):
        """Initialize.

        :param units: size of each output
        :param blocks: number of outputs
        :param kernel_initializer: see tf doc.
        :param bias_initializer: see tf doc.
        :param activation: a tf function.
        :return: a tensor of shape (batch_size, features, blocks)
        """
        # Super
        super().__init__()
        self._n_blocks = blocks
        self._activation = activation
        self._n_units = units

        # Set defaults
        if kernel_initializer is None:
            self._kernel_initializer = tf.initializers.glorot_uniform
        if bias_initializer is None:
            self._bias_initializer = tf.initializers.zeros

    def build(self, input_shape):
        """Instantiate."""
        n_features = input_shape[1]

        # Create parameters
        self.kernel = self.add_weight(
            name="block_kernel",
            shape=(n_features, self._n_units, self._n_blocks),
            dtype=tf.float32,
            initializer=self._kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="block_bias",
            shape=(1, self._n_units, self._n_blocks),
            dtype=tf.float32,
            initializer=self._bias_initializer,
            trainable=True,
        )

    def call(self, inputs):
        """Forward pass."""
        # Duplicate if not partitioned
        assert 2 <= inputs.shape.rank <= 3
        if inputs.shape.rank == 2:
            inputs = tf.tile(tf.expand_dims(inputs, -1), [1, 1, self._n_blocks])
        assert inputs.shape[2] == self._n_blocks

        # Computation
        #   (b: batch, f: features, u: units, k: block)
        inputs = tf.einsum("bfk,fuk->buk", inputs, self.kernel) + self.bias

        # Activation
        if self._activation is not None:
            inputs = self._activation(inputs)

        return inputs
