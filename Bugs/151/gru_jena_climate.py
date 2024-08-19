import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import RMSprop

jena_dir = 'jena_climate_2009_2016.csv'

def data_generate(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    """
    :param data:
    :param lookback: data of back time steps
    :param delay: target of future time steps
    :param min_index:
    :param max_index:
    :param shuffle:
    :param batch_size:
    :param step: the period in time steps
    :return:
    """
    if max_index is None:
        max_index = len(data) - delay -1

    i = min_index + lookback

    while True:
        if shuffle:
            index = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size > max_index:
                i = min_index + lookback
            index = np.arange(i, min(i + batch_size, max_index))
            i += len(index)
        samples = np.zeros(shape=(len(index), lookback // step, data.shape[-1]))
        targets = np.zeros(len(index))

        for j, row in enumerate(index):
            indices = range(index[j] - lookback, index[j], step)
            samples[j] = data[indices]
            targets[j] = data[index[j] + delay][1]
        yield samples, targets

if __name__ == "__main__":
    # Load data
    data = pd.read_csv(jena_dir)
    print(data.shape)  # (420551, 15)
    # Get header
    data_header = data.columns.values
    # print(data_header)  # ['Date Time', 'p (mbar)', ...]

    # Convert DataFrame to Array
    input_data = np.array(data.values[:, 1:], dtype=np.float32)

    # Data preprocessing
    # Normalize data
    mean = np.mean(input_data[:20000], axis=0)
    input_data -= mean
    std = np.std(input_data[:20000], axis=0)
    input_data /= std

    # Data generator parameters
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    train_generate = data_generate(input_data,
                                   lookback=lookback,
                                   delay=delay,
                                   min_index=0,
                                   max_index=200000,
                                   shuffle=True,
                                   step=step,
                                   batch_size=batch_size)
    val_generate = data_generate(input_data,
                                 lookback=lookback,
                                 delay=delay,
                                 min_index=200001,
                                 max_index=300000,
                                 step=step,
                                 batch_size=batch_size)
    test_generate = data_generate(input_data,
                                  lookback=lookback,
                                  delay=delay,
                                  min_index=300001,
                                  max_index=None,
                                  step=step,
                                  batch_size=batch_size)

    # Calculate steps
    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(input_data) - 300001 - lookback) // batch_size

    # Build GRU network
    model = Sequential()
    model.add(GRU(units=32, input_shape=(None, input_data.shape[-1])))
    model.add(Dense(units=1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(train_generate,
                        steps_per_epoch=500,
                        epochs=5,
                        validation_data=val_generate,
                        validation_steps=val_steps)
