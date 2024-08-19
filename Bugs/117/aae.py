# Simple VAE Model example
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
from math import sin, cos, sqrt

LATENT_DIM = 2
DATA_DIM = 784
NUM_LABELS = 10
MODEL_NAME = "aae"

def gaussian(batch_size, n_dim, mean=0, var=1):
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32) * 5.
    return z

class AAE():
    def __init__(self):
        self.encoder, self.decoder, self.disc = self.make_model()

        self.base_lr = 0.001

        self.ae_op = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        self.disc_op = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        self.gen_op = tf.keras.optimizers.Adam(learning_rate=self.base_lr)

    def make_model(self):
        # Encoder
        e_in = tf.keras.layers.Input(shape=(28, 28))
        in_flat = tf.keras.layers.Flatten()(e_in)
        e1 = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_uniform')(in_flat)
        e2 = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_uniform')(e1)
        z = tf.keras.layers.Dense(LATENT_DIM, kernel_initializer='he_uniform')(e2)
        encoder = keras.models.Model(inputs=e_in, outputs=z)
        encoder.summary()

        # Decoder
        d_in = tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1 = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_uniform')(d_in)
        d2 = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer='he_uniform')(d1)
        out = tf.keras.layers.Dense(np.prod((28, 28)), activation='sigmoid', kernel_initializer='he_uniform')(d2)
        out_img = tf.keras.layers.Reshape((28, 28))(out)
        decoder = keras.models.Model(inputs=d_in, outputs=out_img)
        decoder.summary()

        # Discriminator
        z_in = tf.keras.layers.Input(shape=(LATENT_DIM,))
        dc1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(z_in)
        dc2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(dc1)
        dc_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(dc2)
        disc = keras.models.Model(inputs=z_in, outputs=dc_out)
        disc.summary()

        return encoder, decoder, disc

    def disc_latent(self, real_g, fake_g):
        d_real = self.disc(real_g)
        d_fake = self.disc(fake_g)
        return d_real, d_fake

    def update_ae(self, grad):
        enc_vars = self.encoder.trainable_weights
        dec_vars = self.decoder.trainable_weights
        self.ae_op.apply_gradients(zip(grad, enc_vars + dec_vars))

    def update_disc(self, grad):
        disc_vars = self.disc.trainable_weights
        self.disc_op.apply_gradients(zip(grad, disc_vars))

    def update_gen(self, grad):
        enc_vars = self.encoder.trainable_weights
        self.gen_op.apply_gradients(zip(grad, enc_vars))

    def save_model(self, save_path, epoch, batch):
        self.encoder.save(save_path + f'/encoder_{epoch}_{batch}.h5')
        self.decoder.save(save_path + f'/decoder_{epoch}_{batch}.h5')
        self.disc.save(save_path + f'/disc_{epoch}_{batch}.h5')

class MNISTAAE:
    def __init__(self):
        self.net = AAE()

    def train_batch(self, batch, label):
        enc_vars = self.net.encoder.trainable_weights
        dec_vars = self.net.decoder.trainable_weights
        disc_vars = self.net.disc.trainable_weights

        # AE
        with tf.GradientTape() as tape:
            z = self.net.encoder(batch)
            out = self.net.decoder(z)
            recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch, out)) * DATA_DIM
        ae_grads = tape.gradient(recon_loss, enc_vars + dec_vars)
        self.net.update_ae(ae_grads)

        # Disc
        with tf.GradientTape() as tape:
            z = self.net.encoder(batch)
            real_g = gaussian(batch.shape[0], LATENT_DIM)
            d_real, d_fake = self.net.disc_latent(real_g, z)
            dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
            dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
            disc_loss = 0.5 * dc_loss_fake + 0.5 * dc_loss_real
        disc_grads = tape.gradient(disc_loss, disc_vars)
        self.net.update_disc(disc_grads)

        # Gen
        with tf.GradientTape() as tape:
            z = self.net.encoder(batch)
            d_fake = self.net.disc(z)
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))
        gen_grads = tape.gradient(g_loss, enc_vars)
        self.net.update_gen(gen_grads)

        return [recon_loss, disc_loss, g_loss]

    def calc_loss(self, x):
        z = self.net.encoder(x)
        out = self.net.decoder(z)
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, out)) * DATA_DIM

        real_g = gaussian(x.shape[0], LATENT_DIM)
        d_real, d_fake = self.net.disc_latent(real_g, z)
        dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        disc_loss = 0.5 * dc_loss_fake + 0.5 * dc_loss_real

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        return [recon_loss, disc_loss, g_loss]

    def generate_images(self, n=100):
        latents = gaussian(n, LATENT_DIM)
        imgs = self.net.decoder.predict(latents)
        return imgs

    def train(self, num_epochs, batch_size):
        (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.
        x_val = x_val.astype(np.float32) / 255.

        print("Start Training For {} Epochs".format(num_epochs))
        for ep in range(num_epochs):
            batch_iters = int(x_train.shape[0] / batch_size)
            for i in range(batch_iters):
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                batch = x_train[idx]
                label = y_train[idx]
                losses = self.train_batch(batch, label)

                if i % 50 == 0:
                    print(f'Batch {i}, recon: {losses[0]:.5f}, disc: {losses[1]:.5f}, g_loss: {losses[2]:.5f}')

            if (ep + 1) % 10 == 0:
                imgs = self.generate_images()
                print(f"Epoch {ep + 1} completed.")

        self.net.save_model('./' + MODEL_NAME, num_epochs, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST AAE')
    parser.add_argument('--ep', type=int, default=100, help='Num Epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=100, help='Batch Size (default: 100)')
    args = parser.parse_args()

    aae = MNISTAAE()
    aae.train(args.ep, args.batch)
