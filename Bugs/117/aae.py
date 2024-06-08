#Simple VAE Model example
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import *
import argparse
from math import sin,cos,sqrt

LATENT_DIM = 2
DATA_DIM=784
NUM_LABELS=10
MODEL_NAME="aae"

def gaussian(batch_size, n_dim, mean=0, var=1):
    #np.random.seed(0)
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)* 5.
    return z

class AAE():
    def __init__(self):
        self.encoder,self.decoder,self.disc=self.make_model()

        self.base_lr=0.001

        self.ae_op=tf.keras.optimizers.Adam(lr=self.base_lr)
        self.disc_op=tf.keras.optimizers.Adam(lr=self.base_lr)
        self.gen_op=tf.keras.optimizers.Adam(lr=self.base_lr)

    def make_model(self):
        #119 -> 128 -> 64 -> 32 -> 64 -> 128 -> 119
        #Encoder
        e_in=tf.keras.layers.Input(shape=(28,28))
        in_flat=tf.keras.layers.Flatten(input_shape=(28, 28))(e_in)
        e1=tf.keras.layers.Dense(1000, activation='relu',kernel_initializer='he_uniform')(in_flat)
        # e1=tf.keras.layers.BatchNormalization()(e1)
        # e1=tf.keras.layers.Activation('relu')(e1)
        # e1=tf.keras.layers.Dropout(0.5)(e1)
        e2=tf.keras.layers.Dense(1000, activation='relu',kernel_initializer='he_uniform')(e1)
        # e2=tf.keras.layers.BatchNormalization()(e2)
        # e2=tf.keras.layers.Activation('relu')(e2)
        # e2=tf.keras.layers.Dropout(0.5)(e2)
        z=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e2)
        encoder=keras.models.Model(inputs=e_in,outputs=z)
        encoder.summary()

        #Decoder
        d_in=tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1=tf.keras.layers.Dense(1000, activation='relu',kernel_initializer='he_uniform')(d_in)
        # d1=tf.keras.layers.Dropout(0.5)(d1)
        d2=tf.keras.layers.Dense(1000, activation='relu',kernel_initializer='he_uniform')(d1)
        # d2=tf.keras.layers.Dropout(0.5)(d2)
        out=tf.keras.layers.Dense(np.prod((28, 28)), activation='sigmoid',kernel_initializer='he_uniform')(d2)
        out_img=tf.keras.layers.Reshape((28,28))(out)
        decoder=keras.models.Model(inputs=d_in,outputs=out_img)
        decoder.summary()

        #Discriminator
        z_in=tf.keras.layers.Input(shape=(LATENT_DIM,)) #Latent Dist
        # label_in=tf.keras.layers.Input(shape=(1,)) #Latent Dist
        # disc_in=tf.keras.layers.concatenate([z_in, label_in],axis=-1)
        dc1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='he_uniform')(z_in)
        # dc1=tf.keras.layers.Dropout(0.5)(dc1)
        dc2=tf.keras.layers.Dense(256, activation='relu',kernel_initializer='he_uniform')(dc1)
        # dc2=tf.keras.layers.Dropout(0.5)(dc2)
        dc_out=tf.keras.layers.Dense(1, activation='sigmoid',kernel_initializer='he_uniform')(dc2)
        disc=keras.models.Model(inputs=z_in,outputs=dc_out)
        # disc=keras.models.Model(inputs=z_in,outputs=dc_out)
        disc.summary()

        return encoder,decoder,disc

    def disc_latent(self,real_g,fake_g):
        d_real=self.disc(real_g)
        d_fake=self.disc(fake_g)
        return d_real,d_fake

    def update_ae(self,grad):
        enc_vars=self.encoder.trainable_weights
        dec_vars=self.decoder.trainable_weights
        self.ae_op.apply_gradients(zip(grad,enc_vars+dec_vars))

    def update_disc(self,grad):
        disc_vars=self.disc.trainable_weights
        self.disc_op.apply_gradients(zip(grad,disc_vars))

    def update_gen(self,grad):
        enc_vars=self.encoder.trainable_weights
        self.gen_op.apply_gradients(zip(grad,enc_vars))

    def save_model(self,save_path,epoch,batch):
        self.encoder.save(save_path+'/encoder_{}_{}.h5'.format(epoch,batch))
        self.decoder.save(save_path+'/decoder_{}_{}.h5'.format(epoch,batch))
        self.disc.save(save_path+'/disc_{}_{}.h5'.format(epoch,batch))

class MNISTAAE:
    def __init__(self):
        self.net=AAE()

    def train_batch(self,batch,label):
        enc_vars=self.net.encoder.trainable_weights
        dec_vars=self.net.decoder.trainable_weights
        disc_vars=self.net.disc.trainable_weights
        #AE
        with tf.GradientTape() as t:
            t.watch(enc_vars+dec_vars)
            z=self.net.encoder(batch)
            out=self.net.decoder(z)
            #MSE
            # recon_loss = tf.reduce_mean(tf.square(batch - out))
            #BCE
            recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch, out))*DATA_DIM
        ae_grads=t.gradient(recon_loss,enc_vars+dec_vars)
        self.net.update_ae(ae_grads)

        #Disc
        with tf.GradientTape() as t:
            z=self.net.encoder(batch)
            #gaussian(batch_size, n_labels, n_dim, mean=0, var=1, use_label_info=False):
            real_g=gaussian(batch.shape[0],LATENT_DIM)

            d_real,d_fake=self.net.disc_latent(real_g,z)

            #Discriminator Loss
            dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
            dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
            disc_loss = 0.5*dc_loss_fake + 0.5*dc_loss_real

        disc_grads=t.gradient(disc_loss,disc_vars)
        self.net.update_disc(disc_grads)

        #GEN
        with tf.GradientTape() as t:
            z=self.net.encoder(batch)
            d_fake=self.net.disc([z,label])
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        gen_grads=t.gradient(g_loss,enc_vars)
        self.net.update_gen(gen_grads)

        return[recon_loss,disc_loss,g_loss]

    def calc_loss(self,x):
        z=self.net.encoder(x)
        out=self.net.decoder(z)

        #Reconstruction Loss
        #MSE
        # recon_loss = tf.reduce_mean(tf.square(batch - out))
        #BCE
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, out))*DATA_DIM

        #Discriminator Loss
        real_g=gaussian(batch.shape[0],LATENT_DIM)
        d_real,d_fake=self.net.disc_latent(real_g,z)
        dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        disc_loss = 0.5*dc_loss_fake + 0.5*dc_loss_real

        #Generaor Loss
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        return [recon_loss,disc_loss,g_loss]

    def generateImages(self,n=100):
        latents = gaussian(n, LATENT_DIM)
        # latents = 5*np.random.normal(size=(n, LATENT))
        imgs = self.net.decoder.predict(latents)
        return imgs

    def train(self,num_epochs,batch_size):
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.
        # x_test = x_test.astype(np.float32) / 255.
        x_val = x_val.astype(np.float32) / 255.

        model_config={
            'ep': num_epochs,
            'batch_size': batch_size,
            'model_name': MODEL_NAME
        }

        train_loss=[[],[],[]]
        val_loss=[[],[],[]]

        batch_print=50
        print("Start Training For {} Epochs".format(num_epochs))
        for ep in range(num_epochs):
            #np.random.shuffle(x_train)
            batch_iters=int(x_train.shape[0]/batch_size)
            batch_loss=[0,0,0]

            print("\nEpoch {}".format(ep+1))
            for i in range(batch_iters):
                #run batch
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                batch = x_train[idx]
                label = y_train[idx]
                losses=self.train_batch(batch,label)
                for l in range(3):
                    batch_loss[l]+=losses[l]

                if i%batch_print==0:
                    print('Batch loss {} recon:{:.5f}, disc:{:.5f} g_loss:{:.5f}'.format(i,losses[0],losses[1],losses[2]))

            ep_loss=[l/batch_iters for l in batch_loss]
            #Record Train loss
            for l in range(3):
                train_loss[l].append(ep_loss[l])
            print('Epoch loss recon:{:.5f}, disc:{:.5f} g_loss:{:.5f}'.format(ep_loss[0],ep_loss[1],ep_loss[2]))

            #Record Val Loss
            losses=self.calc_loss(x_val)
            for l in range(3):
                val_loss[l].append(losses[l])

            if (ep+1)%10==0:
                #every 10 epochs
                images=self.generateImages()
                imagegrid(images,ep+1,model_config)
                z=self.net.encoder(x_val)
                plot_classes(z,y_val,ep+1,model_config)

            # if ep%100==0:
        self.net.save_model('./'+MODEL_NAME,num_epochs,batch_size)

        #Plot epoch Losses
        #Reconstruction Loss
        plot_losses(train_loss[0],val_loss[0],'recon',model_config)
        #Discriminator Loss
        plot_losses(train_loss[1],val_loss[1],'disc',model_config)
        #Generator Loss
        plot_losses(train_loss[2],val_loss[2],'gen',model_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST AAE')
    parser.add_argument('--ep', type=int, default='100',
                            help='Num Epochs (default: 100)')
    parser.add_argument('--batch', type=int, default='100',
                            help='Batch Size (default: 100)')

    args = parser.parse_args()
    num_epochs=args.ep
    batch_size=args.batch
    aae=MNISTAAE()
    aae.train(num_epochs,batch_size)
