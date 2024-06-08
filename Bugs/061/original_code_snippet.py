import tensorflow as tf

a = tf.constant([[1, 2, 3], [1, 2, 3]])
b = tf.constant([1, 2, 3, 4, 5])

print(tf.concat([a, b], axis=0))