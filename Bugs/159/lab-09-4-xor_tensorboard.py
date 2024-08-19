import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(777)

# Learning rate
learning_rate = 0.01

# Data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_data, y_data, epochs=10001, verbose=0)

# Print final loss and weights
loss, accuracy = model.evaluate(x_data, y_data, verbose=0)
print(f'Final Loss: {loss}')
print(f'Final Accuracy: {accuracy}')

# Predictions
predictions = model.predict(x_data)
print("\nHypothesis: \n", predictions)

# Convert predictions to binary
predicted_classes = (predictions > 0.5).astype(np.float32)
print("\nCorrect: \n", predicted_classes)

# Save accuracy to a file
import json
with open("result.json", "w") as file:
    json.dump({"accuracy": float(accuracy)}, file)

print(f"\nAccuracy saved to result.json: {accuracy}")
