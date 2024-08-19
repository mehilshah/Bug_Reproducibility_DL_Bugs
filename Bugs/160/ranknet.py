import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Input, Lambda
from tensorflow.keras.models import Model

INPUT_DIM = 50

# Define the model
h_1 = Dense(128, activation="relu")
h_2 = Dense(64, activation="relu")
h_3 = Dense(32, activation="relu")
s = Dense(1)

# Relevant document score
rel_doc = Input(shape=(INPUT_DIM,), dtype="float32")
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
rel_score = s(h_3_rel)

# Irrelevant document score
irr_doc = Input(shape=(INPUT_DIM,), dtype="float32")
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
irr_score = s(h_3_irr)

# Subtract scores
negated_irr_score = Lambda(lambda x: -1 * x, output_shape=(1,))(irr_score)
diff = Add()([rel_score, negated_irr_score])

# Pass difference through sigmoid function
prob = Dense(1, activation="sigmoid", weights=[np.array([[1]])], use_bias=False, trainable=False)(diff)

# Build and compile model
model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
model.compile(optimizer="adadelta", loss="binary_crossentropy")

# Fake data
N = 100
X_1 = 2 * np.random.randn(N, INPUT_DIM)
X_2 = np.random.randn(N, INPUT_DIM)
y = np.ones((X_1.shape[0], 1))

# Train model
NUM_EPOCHS = 10
BATCH_SIZE = 10
history = model.fit([X_1, X_2], y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Save loss to file
calculated_loss = history.history['loss'][-1]
import json
with open("result.json", "w") as file:
    json.dump({"loss": float(calculated_loss)}, file)

# Generate scores from document/query features
rel_score_func = tf.keras.backend.function([rel_doc], [rel_score])
rel_scores = rel_score_func([X_1])[0]
irr_scores = rel_score_func([X_2])[0]

print("Relevant Scores:\n", rel_scores)
print("Irrelevant Scores:\n", irr_scores)
