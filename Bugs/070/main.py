import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='features')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(3, activation="softmax", name='classifier')(net)
    return tf.keras.Model(text_input, net)

sentences = tf.constant([
"Improve the physical fitness of your goldfish by getting him a bicycle",
"You are unsure whether or not to trust him but very thankful that you wore a turtle neck",
"Not all people who wander are lost", 
"There is a reason that roses have thorns",
"Charles ate the french fries knowing they would be his last meal",
"He hated that he loved what she hated about hate",
])

labels = tf.random.uniform((6, ), minval=0, maxval=2, dtype=tf.dtypes.int32)

classifier_model = build_classifier_model()
classifier_model.compile(optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=tf.keras.metrics.SparseCategoricalAccuracy())
train_dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))
    
classifier_model.fit(x=train_dataset, epochs=2)