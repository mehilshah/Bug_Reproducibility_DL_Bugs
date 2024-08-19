import tensorflow as tf

# green.shape = TensorShape([64, 40000])
green = tf.ones((64, 40000), dtype='complex64')

tensor = tf.ones(128,1)        # tensor.shape = TensorShape([128])

def mul_and_sum(tensor):
   real = tensor[0:64]
   imag = tensor[64:128]
   complex_tensor = tf.complex(real, imag)
   return tf.reduce_sum((tf.multiply(green, complex_tensor), 1))

res = mul_and_sum(tensor)