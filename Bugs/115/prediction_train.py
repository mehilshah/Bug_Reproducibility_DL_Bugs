import numpy as np
import tensorflow as tf

from absl import app, flags
from prediction_input import build_tfrecord_input
from prediction_model import construct_model

# Constants for intervals and directories
SUMMARY_INTERVAL = 40
VAL_INTERVAL = 200
SAVE_INTERVAL = 2000
DATA_DIR = 'push/push_train'
OUT_DIR = '/tmp/data'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '', 'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 10, 'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1, 'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA', 'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 10, 'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0, 'The k hyperparameter for scheduled sampling, -1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95, 'The percentage of files to use for the training set vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001, 'the base learning rate of the generator')


## Helper functions
def peak_signal_to_noise_ratio(true, pred):
    return 10.0 * tf.math.log(1.0 / mean_squared_error(true, pred)) / tf.math.log(10.0)


def mean_squared_error(true, pred):
    return tf.reduce_sum(tf.square(true - pred)) / tf.cast(tf.size(pred), tf.float32)


class Model(tf.Module):
    def __init__(self, images=None, actions=None, states=None, sequence_length=None):
        self.prefix = prefix = tf.Variable("", dtype=tf.string, trainable=False)
        self.iter_num = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.lr = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)

        if sequence_length is None:
            sequence_length = FLAGS.sequence_length

        # Split into timesteps
        actions = tf.split(actions, num_or_size_splits=actions.shape[1], axis=1)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(states, num_or_size_splits=states.shape[1], axis=1)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(images, num_or_size_splits=images.shape[1], axis=1)
        images = [tf.squeeze(img) for img in images]

        gen_images, gen_states = construct_model(
            images, actions, states, iter_num=self.iter_num,
            k=FLAGS.schedsamp_k, use_state=FLAGS.use_state,
            num_masks=FLAGS.num_masks, cdna=FLAGS.model == 'CDNA',
            dna=FLAGS.model == 'DNA', stp=FLAGS.model == 'STP',
            context_frames=FLAGS.context_frames)

        self.loss, self.psnr_all = self.build_loss(images, states, gen_images, gen_states)

        # Optimizer and training step
        self.train_op = tf.optimizers.Adam(learning_rate=self.lr)

    def build_loss(self, images, states, gen_images, gen_states):
        loss, psnr_all = 0.0, 0.0
        for i, (x, gx) in enumerate(zip(images[FLAGS.context_frames:], gen_images[FLAGS.context_frames - 1:])):
            recon_cost = mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            loss += recon_cost

        for i, (state, gen_state) in enumerate(zip(states[FLAGS.context_frames:], gen_states[FLAGS.context_frames - 1:])):
            state_cost = mean_squared_error(state, gen_state) * 1e-4
            loss += state_cost

        loss /= float(len(images) - FLAGS.context_frames)
        return loss, psnr_all

    @tf.function
    def train_step(self, images, actions, states):
        with tf.GradientTape() as tape:
            loss = self.build_loss(images, actions, states)[0]
        gradients = tape.gradient(loss, self.trainable_variables)
        self.train_op.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


def main(_):
    print('Constructing models and inputs.')
    
    train_images, train_actions, train_states = build_tfrecord_input(training=True)
    val_images, val_actions, val_states = build_tfrecord_input(training=False)
    
    # Create training and validation models
    model = Model(train_images, train_actions, train_states, FLAGS.sequence_length)
    val_model = Model(val_images, val_actions, val_states, FLAGS.sequence_length)

    # Restore pretrained model if provided
    if FLAGS.pretrained_model:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(FLAGS.pretrained_model)

    print('Starting training...')
    
    # Training loop
    for itr in range(FLAGS.num_iterations):
        cost = model.train_step(train_images, train_actions, train_states)
        print(f"Iteration {itr}: Loss: {cost}")

        if itr % VAL_INTERVAL == 0:
            val_loss = val_model.train_step(val_images, val_actions, val_states)
            print(f"Validation Loss at iteration {itr}: {val_loss}")

        if itr % SAVE_INTERVAL == 0:
            checkpoint.save(file_prefix=f"{FLAGS.output_dir}/ckpt_{itr}")
            print(f"Checkpoint saved at iteration {itr}.")

    print("Training complete")


if __name__ == '__main__':
    app.run(main)
