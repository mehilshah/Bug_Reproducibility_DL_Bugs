import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

class GibbsPrunedConv2D(layers.Conv2D):
    def __init__(self, filters, kernel_size, p=0.5, hamiltonian='unstructured',
                 c=1.0, train_pruning_mode='gibbs', mcmc_steps=20, **kwargs):
        self.p = p
        self.hamiltonian = hamiltonian
        self.c = c
        self.train_pruning_mode = train_pruning_mode
        self.mcmc_steps = mcmc_steps
        self.beta = tf.Variable(1.0, trainable=False, name='beta')
        super().__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.mask = tf.zeros_like(self.kernel)

    def call(self, inputs, training=None):
        call_input_shape = inputs.shape
        recreate_conv_op = call_input_shape[1:] != self._build_input_shape[1:]

        if recreate_conv_op:
            self._convolution_op = self._get_convolution_op(call_input_shape)

        mask = tf.cond(tf.cast(training, tf.bool), lambda: self.train_mask(), lambda: self.test_mask())
        self.add_metric(1 - tf.reduce_mean(mask), name='gp_mask_p', aggregation='mean')
        outputs = self._convolution_op(inputs, self.kernel * mask)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self._data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def train_mask(self):
        W2 = self.kernel ** 2
        if self.train_pruning_mode == 'gibbs':
            return self.test_mask()
        elif self.train_pruning_mode == 'kernel':
            kernel_sums = tf.reduce_sum(W2, axis=[0, 1])
            Qp = tfp.stats.percentile(kernel_sums, self.p * 100, interpolation='linear')
            return tf.cast(kernel_sums >= Qp, tf.float32)[None, None, :, :]
        elif self.train_pruning_mode == 'filter':
            filter_sums = tf.reduce_sum(W2, axis=[0, 1, 2])
            Qp = tfp.stats.percentile(filter_sums, self.p * 100, interpolation='linear')
            return tf.cast(filter_sums >= Qp, tf.float32)[None, None, None, :]
        else:
            raise ValueError("train_pruning_mode must be one of 'gibbs', 'kernel', or 'filter'")

    def test_mask(self):
        W2 = self.kernel ** 2
        Qp = tfp.stats.percentile(tf.reshape(W2, [-1]), self.p * 100)
        if self.hamiltonian == 'unstructured':
            P0 = 1 / (1 + tf.exp(self.beta * (W2 - Qp)))
            R = tf.random.uniform(tf.shape(P0))
            return tf.cast(R > P0, tf.float32)

    def get_config(self):
        config = {
            'p': self.p,
            'hamiltonian': self.hamiltonian,
            'c': self.c,
            'train_pruning_mode': self.train_pruning_mode,
            'mcmc_steps': self.mcmc_steps,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def set_beta(self, beta):
        self.beta.assign(beta)

class GibbsPruningAnnealer(keras.callbacks.Callback):
    def __init__(self, beta_schedule, verbose=0):
        super().__init__()
        self.beta_schedule = beta_schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        beta = self.beta_schedule[epoch] if epoch < len(self.beta_schedule) else self.beta_schedule[-1]
        count = 0
        for layer in self.model.layers:
            if isinstance(layer, GibbsPrunedConv2D):
                layer.set_beta(beta)
                count += 1
        if self.verbose > 0:
            print(f'GibbsPruningAnnealer: set beta to {beta} in {count} layers')