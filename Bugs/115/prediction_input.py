import os
import numpy as np
import tensorflow as tf

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Dimension of the state and action.
STATE_DIM = 5

class InputPipeline:
    def __init__(self, data_dir, sequence_length, batch_size, train_val_split=0.8, use_state=True):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.use_state = use_state

    def _parse_function(self, serialized_example):
        image_seq, state_seq, action_seq = [], [], []

        for i in range(self.sequence_length):
            image_name = f'move/{i}/image/encoded'
            action_name = f'move/{i}/commanded_pose/vec_pitch_yaw'
            state_name = f'move/{i}/endeffector/vec_pitch_yaw'
            
            features = {
                image_name: tf.io.FixedLenFeature([], tf.string)
            }
            
            if self.use_state:
                features[action_name] = tf.io.FixedLenFeature([STATE_DIM], tf.float32)
                features[state_name] = tf.io.FixedLenFeature([STATE_DIM], tf.float32)
            
            parsed_features = tf.io.parse_single_example(serialized_example, features)
            
            # Decode the image
            image = tf.io.decode_jpeg(parsed_features[image_name], channels=COLOR_CHAN)
            image = tf.image.resize_with_crop_or_pad(image, ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
            image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
            image = tf.cast(image, tf.float32) / 255.0
            image_seq.append(image)
            
            if self.use_state:
                state = parsed_features[state_name]
                action = parsed_features[action_name]
                state_seq.append(state)
                action_seq.append(action)
        
        image_seq = tf.stack(image_seq, axis=0)
        
        if self.use_state:
            state_seq = tf.stack(state_seq, axis=0)
            action_seq = tf.stack(action_seq, axis=0)
            return image_seq, action_seq, state_seq
        else:
            zeros_seq = tf.zeros([self.sequence_length, STATE_DIM])
            return image_seq, zeros_seq, zeros_seq

    def build_dataset(self, training=True):
        # Get all filenames
        filenames = tf.io.gfile.glob(os.path.join(self.data_dir, '*'))
        
        if not filenames:
            raise RuntimeError('No data files found.')

        # Split filenames into training and validation
        index = int(np.floor(self.train_val_split * len(filenames)))
        if training:
            filenames = filenames[:index]
        else:
            filenames = filenames[index:]

        # Build a dataset from the filenames
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch the dataset
        if training:
            dataset = dataset.shuffle(buffer_size=100 * self.batch_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset


# Example usage
if __name__ == "__main__":
    data_dir = "/path/to/tfrecords"
    sequence_length = 10
    batch_size = 32
    train_val_split = 0.8
    use_state = True

    pipeline = InputPipeline(data_dir, sequence_length, batch_size, train_val_split, use_state)

    # Build the training dataset
    train_dataset = pipeline.build_dataset(training=True)

    # Iterate over the dataset
    for image_batch, action_batch, state_batch in train_dataset:
        print("Image batch shape:", image_batch.shape)
        print("Action batch shape:", action_batch.shape)
        print("State batch shape:", state_batch.shape)
