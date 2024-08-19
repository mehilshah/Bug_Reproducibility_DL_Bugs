import numpy as np
import tensorflow as tf

x1 = np.asarray([1,0,0])
x2 = np.asarray([0,1,0])
x3 = np.asarray([0,0,1])

group_a = np.stack([x1,x2])
group_b = np.stack([x3])
ac = tf.ragged.stack([group_a,group_b], axis=0)

print(ac.shape)
