green_mat = sio.loadmat('green.mat')
green = np.array(green_mat['G2'])
green = tf.convert_to_tensor(green)
green = tf.cast(green, dtype='complex64')  # >>>green.shape = TensorShape([64, 40000])



tensor = tf.ones(128,1)        # tensor.shape = TensorShape([128])

def mul_and_sum(tensor):
   real = tensor[0:64]
   imag = tensor[64:128]
   complex_tensor = tf.complex(real, imag)
   return tf.reduce_sum((tf.multiply(green, complex_tensor), 1))

res = mul_and_sum(tensor)