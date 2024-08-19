import tensorflow as tf
import numpy as np

def compute_loss(pred, actual):
    return tf.reduce_mean(tf.square(tf.subtract(pred, actual)))

def compute_gradient(model, pred, actual):
    """compute gradients with given noise and input"""
    with tf.GradientTape() as tape:
        loss = compute_loss(pred, actual)
    grads = tape.gradient(loss, model.trainable_variables)
    return grads, loss

def apply_gradients(optimizer, grads, model_vars):
    optimizer.apply_gradients(zip(grads, model_vars))

def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

model = make_model()
optimizer = tf.keras.optimizers.Adam(1e-4)

x = np.linspace(0,1,1000)
y = x + np.random.normal(0,0.3,1000)
y = y.astype('float32')
train_dataset = tf.data.Dataset.from_tensor_slices((y.reshape(-1,1)))

epochs = 2
batch_size = 25
itr = y.shape[0] // batch_size
for epoch in range(epochs):
    for data in train_dataset.batch(batch_size):
        preds = model(data)
        grads, loss = compute_gradient(model, preds, data)
        apply_gradients(optimizer, grads, model.trainable_variables)

# Gradient output: [None, None, None, None, None, None]