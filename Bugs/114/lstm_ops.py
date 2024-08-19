import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class BasicConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, num_channels, filter_size=5, forget_bias=1.0, **kwargs):
        super(BasicConvLSTMCell, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.forget_bias = forget_bias
        self.conv = Conv2D(4 * num_channels, (filter_size, filter_size), padding='same')

    def build(self, input_shape):
        # Build the layer based on the input shape
        self.built = True

    def call(self, inputs, states):
        h, c = tf.split(states, num_or_size_splits=2, axis=3)

        # Concatenate input and hidden state
        inputs_h = tf.concat([inputs, h], axis=3)
        
        # Compute the gates in one convolutional operation
        i_j_f_o = self.conv(inputs_h)

        # Split the convolution results into the 4 LSTM gates
        i, j, f, o = tf.split(i_j_f_o, num_or_size_splits=4, axis=3)

        # Compute the new cell and hidden states
        new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat([new_c, new_h], axis=3)

def init_state(inputs, state_shape, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
    """Helper function to create an initial state given inputs."""
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    else:
        batch_size = 0

    initial_state = state_initializer(shape=[batch_size] + state_shape, dtype=dtype)

    return initial_state

# Example usage:
if __name__ == "__main__":
    # Assume some input tensor shape: batch_size, height, width, channels
    inputs = tf.random.normal([8, 64, 64, 3])

    # Initializing the ConvLSTM cell
    num_channels = 32
    conv_lstm_cell = BasicConvLSTMCell(num_channels=num_channels)

    # Initial state
    state_shape = [64, 64, num_channels * 2]
    initial_state = init_state(inputs, state_shape)

    # Forward pass (basic call with inputs and initial state)
    outputs, new_state = conv_lstm_cell(inputs, initial_state)

    print("Outputs shape:", outputs.shape)
    print("New state shape:", new_state.shape)
