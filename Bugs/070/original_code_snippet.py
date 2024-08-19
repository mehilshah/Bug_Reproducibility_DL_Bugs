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

corpus=df_speech_EN_merged["contents"]
corpus.shape
labels=np.asarray(df_speech_EN_merged["Classes"].astype("int"))
labels.shape

train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        {
            "features":tf.cast(corpus_train.values, tf.string),
            "labels":tf.cast(labels_train, tf.int32) #labels is already an array, no need for .values
        }
    )
test_dataset = tf.data.Dataset.from_tensor_slices(
    {"features":tf.cast(corpus_test.values, tf.string),
     "labels":tf.cast(labels_test, tf.int32)
    } #labels is already an array, no need for .values
    )
)

classifier_model.fit(x=train_dataset,
                               validation_data=test_dataset,
                               epochs=2)