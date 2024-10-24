import configparser
import pickle
from datetime import datetime

import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam


def preprocess(processing_messages):
    sequences = []
    predictions = []
    for i in range(0, len(processing_messages) - sentence_length, overlapping_step):
        sequences.append(processing_messages[i: i + sentence_length])
        predictions.append(processing_messages[i + sentence_length])

    x = np.zeros((len(sequences), sentence_length, len(characters)), dtype=np.bool)
    y = np.zeros((len(sequences), len(characters)), dtype=np.bool)

    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            x[i, t, character_map[char]] = 1
        y[i, character_map[predictions[i]]] = 1
    return x, y


# Loading configuration
config = configparser.ConfigParser()
config.read('config.ini')

processing_percentage = config.getint('APP', 'PROCESSING_PERCENTAGE')
sentence_length = config.getint('LEARNING', 'SENTENCE_LENGTH')

overlapping_step = 3
batch_size = 128
epochs = 5

rnn_layers = [1, 2, 3]
rnn_node_sizes = [128, 256, 512]
dense_layers = [0, 1]
dense_node_sizes = [128]

# Prepare sequences and predictions
file = open(f'data/processed_{processing_percentage}.pickle', 'rb')
connected_messages = pickle.load(file)
characters = sorted(list(set(connected_messages)))
character_map = dict((c, i) for i, c in enumerate(characters))
indicator_map = dict((i, c) for i, c in enumerate(characters))

divider_index = int(len(connected_messages) * 0.8)
training_messages = connected_messages[:divider_index]
validation_messages = connected_messages[divider_index:]

train_x, train_y = preprocess(training_messages)
validation_x, validation_y = preprocess(validation_messages)

history = 0

# Modeling
for rnn_layer in rnn_layers:
    for rnn_node_size in rnn_node_sizes:
        for dense_layer in dense_layers:
            for dense_node_size in dense_node_sizes:
                name = f'{rnn_layer}-{rnn_node_size}-rnn-{dense_layer}-{dense_node_size}-dense'
                name = f'{name}-{processing_percentage}-proc-{sentence_length}-len-{overlapping_step}-lap'
                name = f'{name}-{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}'
                print(f'Current computation: {name}')

                model = Sequential()

                model.add(LSTM(
                    rnn_node_size,
                    input_shape=(sentence_length, len(characters)),
                    return_sequences=rnn_layer > 1)
                )
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                for layer in range(1, rnn_layer):
                    model.add(LSTM(rnn_node_size, return_sequences=layer < rnn_layer - 1))
                    model.add(Dropout(0.1))
                    model.add(BatchNormalization())

                for _ in range(dense_layer):
                    model.add(Dense(dense_node_size, activation='relu'))
                    model.add(Dropout(0.2))

                model.add(Dense(len(characters), activation='softmax'))

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.001, decay=1e-6),  # RMSprop(lr=0.01)
                    metrics=['accuracy']
                )

                checkpoint = ModelCheckpoint(
                    f'models/{name}.model',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
                log_dir = f'logs/{name}'
                tensorboard = TensorBoard(log_dir=log_dir)

                history = model.fit(
                    train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(validation_x, validation_y),
                    callbacks=[checkpoint, tensorboard]
                )
              
import json
train_loss = history.history["loss"]
loss = train_loss[-1:]
file = open(file="result.json", mode="w")  
model_loss = np.float64(loss)
res = {"loss" : model_loss}
json.dump(res, file)
file.close()
