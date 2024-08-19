import numpy as np
import tensorflow as tf
import json

def generate_field(shape, alive_prob=0.5):
    """Generates a random field with cells being alive with a certain probability."""
    return (np.random.rand(*shape) < alive_prob).astype(np.int32)

def update_field(field, hw_axis=(1, 2)):
    """
    Updates the state of the field according to the Game of Life rules.

    Args:
        field: ndarray of shape (N, H, W, 1), where N is the number of samples.
        hw_axis: tuple of height and width axes.

    Returns:
        ndarray: Updated field of the same shape.
    """
    h_ax, w_ax = hw_axis
    neighbours = np.zeros(field.shape, dtype=np.int32)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not (dx or dy):
                continue
            neighbours += np.roll(np.roll(field, dx, w_ax), dy, h_ax)

    return np.logical_or(
        neighbours == 3,
        np.logical_and(field, neighbours == 2)
    ).astype(np.int32)

def pad_field(field):
    """
    Pads the height and width dimensions of the field to emulate cyclic boundaries.

    Args:
        field: ndarray of shape (N, H, W, 1).

    Returns:
        ndarray: Padded field of shape (N, H+2, W+2, 1).
    """
    return np.pad(field, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='wrap')

def generate_dataset(num_samples, height, width, alive_prob=0.5):
    """
    Generates a dataset of fields and their next states.

    Args:
        num_samples: Number of samples to generate.
        height: Height of the field.
        width: Width of the field.
        alive_prob: Probability of a cell being alive.

    Returns:
        Tuple of (X, y) where X is the field states and y is the next states.
    """
    shape = (num_samples, height, width, 1)
    X = generate_field(shape, alive_prob)
    y = update_field(X)
    return X, y

def life_nn(height, width, num_filters=10, num_channels=20, loss='binary_crossentropy', optimizer='adam'):
    """
    Creates a CNN model for the Game of Life.

    Args:
        height: Height of the input field.
        width: Width of the input field.
        num_filters: Number of filters in the first convolution layer.
        num_channels: Number of channels in the second convolution layer.
        loss: Loss function.
        optimizer: Optimizer.

    Returns:
        A compiled Keras model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation='relu',
            input_shape=(height + 2, width + 2, 1)
        ),
        tf.keras.layers.Conv2D(
            num_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation='relu'
        ),
        tf.keras.layers.Conv2D(
            1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation='sigmoid'
        )
    ])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def train(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=1):
    """
    Trains the neural network model on the training and validation sets.

    Args:
        model: The Keras model to train.
        X_train: Training input data.
        y_train: Training labels.
        X_val: Validation input data.
        y_val: Validation labels.
        batch_size: Batch size for training.
        epochs: Number of epochs to train for.
    """
    X_train_padded = pad_field(X_train)
    X_val_padded = pad_field(X_val)
    model.fit(X_train_padded, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_padded, y_val)
    )

def evaluate(model, X_test, y_test, batch_size=64):
    """
    Evaluates the model's performance on the test set.

    Args:
        model: The Keras model to evaluate.
        X_test: Test input data.
        y_test: Test labels.
        batch_size: Batch size for evaluation.

    Returns:
        Tuple of loss and accuracy.
    """
    X_test_padded = pad_field(X_test)
    return model.evaluate(X_test_padded, y_test, batch_size=batch_size, verbose=0)

def evaluate_prob_grid(model):
    """
    Evaluates model performance across different cell alive probabilities.

    Args:
        model: The Keras model to evaluate.
    """
    total_loss = 0
    total_acc = 0
    counter = 0
    shape = model.input_shape
    height, width = shape[1] - 2, shape[2] - 2

    for alive_prob in np.linspace(0.1, 0.9, 9):
        X_test, y_test = generate_dataset(1000, height, width, alive_prob)
        loss, acc = evaluate(model, X_test, y_test)
        counter += 1
        total_acc += acc
        total_loss += loss
        print(f'P_alive={alive_prob:.1f} Loss:{loss:.2f} Acc:{acc:.2f}')

    average_loss = total_loss / counter
    with open("result.json", "w") as file:
        json.dump({"loss": average_loss}, file)

def print_evolution(height, width, alive_prob=0.5, epochs=20):
    """
    Visualizes the field evolution for debugging purposes.

    Args:
        height: Height of the field.
        width: Width of the field.
        alive_prob: Probability of a cell being alive.
        epochs: Number of epochs to run.
    """
    import time

    field = generate_field((height, width))

    def print_field(field):
        for row in field:
            print(''.join('X' if x else '.' for x in row))

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        print_field(field)
        print()
        field = update_field(field, hw_axis=(0, 1))
        time.sleep(0.3)

if __name__ == '__main__':
    import sys
    height = int(sys.argv[1])
    width = int(sys.argv[2])

    print('Building model:')
    model = life_nn(height, width)
    model.summary()

    num_train_samples = 15000
    num_val_samples = 3000
    X_train, y_train = generate_dataset(num_train_samples, height, width)
    X_val, y_val = generate_dataset(num_val_samples, height, width)

    print('Training model:')
    train(model, X_train, y_train, X_val, y_val, epochs=5)

    print('Evaluating model:')
    evaluate_prob_grid(model)
