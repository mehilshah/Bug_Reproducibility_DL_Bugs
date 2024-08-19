pred_numpy = numpy.asarray([y_pred], numpy.int32)
act_numpy = numpy.asarray([y_act], numpy.int32)

# compute multiclass f1
metric = tfa.metrics.F1Score(num_classes=num_classes, average="macro")
metric.update_state(act_numpy, pred_numpy)
print(metric.result().numpy())