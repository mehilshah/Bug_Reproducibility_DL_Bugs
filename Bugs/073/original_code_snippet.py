from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import  BertConfig
import tensorflow as tf

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
robertamodel = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=7)
print('\nBert Model',robertamodel.summary())
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('0accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5,epsilon=1e-08)

robertamodel.compile(loss=loss,optimizer=optimizer,metrics=[metric])
print(robertamodel.summary())