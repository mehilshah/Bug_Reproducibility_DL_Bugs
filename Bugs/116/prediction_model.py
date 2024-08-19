import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# Kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5

def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    use_state=True,
                    num_masks=10,
                    stp=False,
                    cdna=True,
                    dna=False,
                    context_frames=2):
    """Build convolutional LSTM video predictor using STP, CDNA, or DNA."""
    
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].shape
    lstm_func = basic_conv_lstm_cell

    # Generated robot states and images.
    gen_states, gen_images = [], []
    current_state = states[0]

    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        num_ground_truth = tf.cast(
            tf.round(tf.cast(batch_size, tf.float32) * (k / (k + tf.exp(iter_num / k)))), 
            tf.int32
        )
        feedself = False

    # LSTM state sizes and states.
    lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
    lstm_states = [None] * 7

    for image, action in zip(images[:-1], actions[:-1]):
        reuse = bool(gen_images)
        done_warm_start = len(gen_images) > context_frames - 1

        if feedself and done_warm_start:
            prev_image = gen_images[-1]
        elif done_warm_start:
            prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
        else:
            prev_image = image

        # Concatenate action and state
        state_action = tf.concat([action, current_state], axis=1)

        # Encoder
        enc0 = layers.Conv2D(32, [5, 5], strides=2, padding='same', activation=None)(prev_image)
        enc0 = tf.keras.layers.LayerNormalization()(enc0)

        # LSTM layers
        hidden_states = []
        hidden_states.append(lstm_func(enc0, lstm_states[0], lstm_size[0], 'state1'))
        for i in range(1, 7):
            hidden = tf.keras.layers.LayerNormalization()(hidden_states[-1])
            hidden = layers.Conv2D(lstm_size[i], [3, 3], strides=2, padding='same')(hidden)
            hidden_states.append(lstm_func(hidden, lstm_states[i], lstm_size[i], f'state{i + 1}'))
        
        # Decoder
        hidden5 = layers.Conv2DTranspose(lstm_size[4], 3, strides=2, padding='same')(hidden_states[4])
        hidden6 = layers.Conv2DTranspose(lstm_size[5], 3, strides=2, padding='same')(hidden5)
        hidden6 = tf.concat([hidden6, hidden_states[3]], axis=-1)  # skip connection

        hidden7 = layers.Conv2DTranspose(lstm_size[6], 3, strides=2, padding='same')(hidden6)
        hidden7 = tf.concat([hidden7, enc0], axis=-1)  # skip connection

        enc6 = layers.Conv2DTranspose(color_channels, 3, strides=2, padding='same')(hidden7)

        # Use CDNA for transformation
        if cdna:
            transformed = cdna_transformation(prev_image, hidden_states[-1], num_masks, color_channels)
        
        # Add masks
        masks = layers.Conv2DTranspose(num_masks + 1, 1, strides=1, padding='same')(enc6)
        masks = tf.nn.softmax(masks, axis=-1)
        mask_list = tf.split(masks, num_masks + 1, axis=-1)

        output = mask_list[0] * prev_image
        for layer, mask in zip(transformed, mask_list[1:]):
            output += layer * mask
        
        gen_images.append(output)

        # Predict next state
        current_state = layers.Dense(current_state.shape[-1], activation=None)(state_action)
        gen_states.append(current_state)

    return gen_images, gen_states


# The LSTM cell is changed to Keras layers.
class basic_conv_lstm_cell(layers.Layer):
    """A basic Conv LSTM cell implementation in Keras."""
    
    def __init__(self, num_units, kernel_size, strides=(1, 1), padding='same', **kwargs):
        super(basic_conv_lstm_cell, self).__init__(**kwargs)
        self.conv_lstm = layers.ConvLSTM2D(
            num_units, 
            kernel_size, 
            strides=strides, 
            padding=padding, 
            return_sequences=True, 
            return_state=True
        )

    def call(self, inputs, states):
        return self.conv_lstm(inputs, initial_state=states)


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels):
    """Apply convolutional dynamic neural advection to previous image."""
    
    batch_size = cdna_input.shape[0]
    
    cdna_kerns = layers.Dense(
        DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks, activation=None)(cdna_input)
    
    # Reshape and normalize.
    cdna_kerns = tf.reshape(cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
    cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
    norm_factor = tf.reduce_sum(cdna_kerns, axis=[1, 2, 3], keepdims=True)
    cdna_kerns /= norm_factor

    # Apply transformation
    transformed = []
    for kernel in tf.split(cdna_kerns, batch_size, axis=0):
        kernel = tf.squeeze(kernel, axis=0)
        transformed.append(tf.nn.depthwise_conv2d(prev_image, kernel, [1, 1, 1, 1], 'SAME'))
    
    return transformed


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data points."""
    idx = tf.random.shuffle(tf.range(batch_size))
    ground_truth_idx = idx[:num_ground_truth]
    generated_idx = idx[num_ground_truth:]

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)

    return tf.dynamic_stitch([ground_truth_idx, generated_idx], [ground_truth_examps, generated_examps])

