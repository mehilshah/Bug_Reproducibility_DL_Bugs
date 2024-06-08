embedding_size=200
UNITS=128

encoder_inputs = Input(shape=(None,), name="encoder_inputs")

encoder_embs=Embedding(num_encoder_tokens, embedding_size, name="encoder_embs")(encoder_inputs)

#encoder lstm
encoder = LSTM(UNITS, return_state=True, name="encoder_LSTM") #(encoder_embs)
encoder_outputs, state_h, state_c = encoder(encoder_embs)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,), name="decoder_inputs")
decoder_embs = Embedding(num_decoder_tokens, embedding_size, name="decoder_embs")(decoder_inputs)

#decoder lstm
decoder_lstm = LSTM(UNITS, return_sequences=True, return_state=True, name="decoder_LSTM")
decoder_outputs, _, _ = decoder_lstm(decoder_embs, initial_state=encoder_states)

attention=Attention(name="attention_layer")
attention_out=attention([encoder_outputs, decoder_outputs])

decoder_concatenate=Concatenate(axis=-1, name="concat_layer")([decoder_outputs, attention_out])
decoder_outputs = TimeDistributed(Dense(units=num_decoder_tokens, 
                                  activation='softmax', name="decoder_denseoutput"))(decoder_concatenate)

model=Model([encoder_inputs, decoder_inputs], decoder_outputs, name="s2s_model")
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit([encoder_training_input, decoder_training_input], decoder_training_target,
      epochs=100,
      batch_size=32,
      validation_split=0.2,)