inputs = keras.Input(shape=(64,64,1), dtype='float32')

x = keras.layers.Conv2D(12,(9,9), padding="same",input_shape=(64,64,1), dtype='float32',activation='relu')(inputs)
x = keras.layers.Conv2D(18,(7,7), padding="same", activation='relu')(x)

x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
x = keras.layers.Dropout(0.25)(x)

x = keras.layers.Dense(50, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
     optimizer=keras.optimizers.Adam(0.01),
      metrics=["acc"],
      )
model.fit(x_train, y_train, batch_size=32, epochs = 20, validation_split= 0.3,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])