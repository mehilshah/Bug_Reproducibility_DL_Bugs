import tensorflow as tf

X = tf.ragged.constant([[0,1,2], [0,1]])
def outer_product(x):
  return x[...,None]*x[None,...]
tf.map_fn(outer_product, X)