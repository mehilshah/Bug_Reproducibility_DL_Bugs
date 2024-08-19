import tensorflow as tf
import tensorflow_addons as tfa
import numpy
import random

y_pred = random.randint(0, 1 )
y_act = random.randint(0, 1)

pred_numpy = numpy.asarray([y_pred], numpy.int32)
act_numpy = numpy.asarray([y_act], numpy.int32)

# compute multiclass f1
metric = tfa.metrics.F1Score(num_classes=2, average="macro")
metric.update_state(act_numpy, pred_numpy)
print(metric.result().numpy())