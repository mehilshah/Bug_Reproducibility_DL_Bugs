import tensorflow as tf
import numpy as np
import time

a = np.ones((9000,4000))
b = np.ones((4000,9000))

a2 = [a,a,a,a,a,a,a]
b2 = [b,b,b,b,b,b,b]

a3 = np.ones((7,9000,4000))
b3 = np.ones((7,4000,9000))

with tf.device('/gpu:0'):
    
    # first multiplication

    a2 = tf.convert_to_tensor(a)
    b2 = tf.convert_to_tensor(b)

    start = time.time()
    c = tf.matmul([b2,b2,b2,b2,b2,b2,b2], [a2,a2,a2,a2,a2,a2,a2])
    print("first multiplication time: ", time.time() - start)
    del c, a2, b2

    # second multiplication

    a3 = tf.convert_to_tensor(a3)
    b3 = tf.convert_to_tensor(b3)

    start = time.time()
    c = tf.matmul(b3, a3)
    print("second multiplication time: ", time.time() - start)
    del c, a3, b3

    # third multiplication

    start = time.time()
    n = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='n')
    m = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='m')
    print("constant init time: ",time.time() - start)

    c = tf.matmul([n,n], [m,m])
    print("constant init plus third multiplication time: ", time.time() - start)