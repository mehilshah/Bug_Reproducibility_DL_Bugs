x = x.reshape(-1, 28, 28, 1)
model = Sequential([
    Conv2D(8, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)),
    Dense(64, activation=tf.nn.relu),
    Dense(64, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x, y, epochs=5)