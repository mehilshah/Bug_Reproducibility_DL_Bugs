import tensorflow as tf

x = tf.range(0,64*5)
x = tf.reshape(x, [1,5, 64])

y = tf.range(0,5)
y = tf.reshape(y, [1, 5])

prodct = x*y