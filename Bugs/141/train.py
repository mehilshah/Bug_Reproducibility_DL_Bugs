import argparse
import tensorflow as tf
import mlflow
import mlflow.keras
import mlflow.pyfunc
from mlflow.pyfunc import PythonModel
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Train a Keras CNN model for MNIST classification')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=4)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
num_classes = 10

mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)

# Input image dimensions
img_rows, img_cols = 28, 28

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)

mlflow.log_metric("cross_entropy_test_loss", score[0])
mlflow.log_metric("test_accuracy", score[1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Log the Keras model
mlflow.keras.log_model(model, artifact_path="keras-model")

# Define Conda environment
conda_env = _mlflow_conda_env(
    additional_conda_deps=[
        "tensorflow=={}".format(tf.__version__),
    ],
    additional_pip_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__),
        "mlflow=={}".format(mlflow.__version__),
    ]
)

class KerasMnistCNN(PythonModel):
    def load_context(self, context):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.keras.backend.set_learning_phase(0)
            self.model = mlflow.keras.load_model(context.artifacts["keras-model"])

    def predict(self, context, input_df):
        with self.graph.as_default():
            return self.model.predict(input_df.values.reshape(-1, 28, 28, 1))

# Log the PyFunc model
mlflow.pyfunc.log_model(
    artifact_path="keras-pyfunc",
    python_model=KerasMnistCNN(),
    artifacts={
        "keras-model": mlflow.get_artifact_uri("keras-model")
    },
    conda_env=conda_env
)

print(mlflow.active_run().info.run_uuid)
