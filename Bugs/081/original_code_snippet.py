# %%
# THIS CODE CELL LOADS THE PACKAGES USED IN THIS NOTEBOOK

# Load core packages for data analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt

# Load deep learning packages
import tensorflow as tf
from tensorflow.keras.datasets.cifar100 import load_data      
from tensorflow.keras import (Model, layers)          
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import (to_categorical, plot_model)
from tensorflow.lookup import (StaticHashTable, KeyValueTensorInitializer)

# Print versions of main ML packages
print("Tensorflow version " + tf.__version__)

# %%
# THIS CODE CELL LOADS DATASETS AND CHECKS DATA DIMENSIONS

# There is an option to load the "fine" (100 fine classes) or "coarse" (20 super classes) labels with integer (int) encodings
# We will load both labels for hierarchical classification tasks
(x_train, y_train_fine_int), (x_test, y_test_fine_int) = load_data(label_mode="fine")
(_, y_train_coarse_int), (_, y_test_coarse_int) = load_data(label_mode="coarse")

# EXTRACT DATASET PARAMETERS FOR USE LATER ON
num_fine_classes = 100
num_coarse_classes = 20
input_shape = x_train.shape[1:]     

# DEFINE BATCH SIZE
batch_size = 50

# %%
# THIS CODE CELL PROVIDES THE CODE TO LINK INTEGER LABELS TO MEANINGFUL WORD LABELS
# Fine and coarse labels are provided as integers.  We will want to link them both to meaningful world labels.

# CREATE A DICTIONARY TO MAP THE 20 COARSE LABELS TO THE 100 FINE LABELS

# This mapping comes from https://keras.io/api/datasets/cifar100/ 
# Except "computer keyboard" should just be "keyboard" for the encoding to work
CoarseLabels_to_FineLabels = {
    "aquatic mammals":                  ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish":                             ["aquarium fish", "flatfish", "ray", "shark", "trout"],
    "flowers":                          ["orchids", "poppies", "roses", "sunflowers", "tulips"],
    "food containers":                  ["bottles", "bowls", "cans", "cups", "plates"],
    "fruit and vegetables":             ["apples", "mushrooms", "oranges", "pears", "sweet peppers"],
    "household electrical devices":     ["clock", "keyboard", "lamp", "telephone", "television"],
    "household furniture":              ["bed", "chair", "couch", "table", "wardrobe"],
    "insects":                          ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores":                 ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things":    ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes":     ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores":   ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals":             ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates":         ["crab", "lobster", "snail", "spider", "worm"],
    "people":                           ["baby", "boy", "girl", "man", "woman"],
    "reptiles":                         ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals":                    ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees":                            ["maple", "oak", "palm", "pine", "willow"],
    "vehicles 1":                       ["bicycle", "bus", "motorcycle", "pickup" "truck", "train"],
    "vehicles 2":                       ["lawn-mower", "rocket", "streetcar", "tank", "tractor"]
}

# CREATE A DICTIONARY TO MAP THE INTEGER-ENCODED COARSE LABEL TO THE WORD LABEL
# Create list of Course Labels
CoarseLabels = list(CoarseLabels_to_FineLabels.keys())

# The target variable in CIFER100 is encoded such that the coarse class is assigned an integer based on its alphabetical order
# The CoarseLabels list is already alphabetized, so no need to sort
CoarseInts_to_CoarseLabels = dict(enumerate(CoarseLabels))

# CREATE A DICTIONARY TO MAP THE WORD LABEL TO THE INTEGER-ENCODED COARSE LABEL
CoarseLabels_to_CoarseInts = dict(zip(CoarseLabels, range(20)))


# CREATE A DICTIONARY TO MAP THE 100 FINE LABELS TO THE 20 COARSE LABELS
FineLabels_to_CoarseLabels = {}
for CoarseLabel in CoarseLabels:
    for FineLabel in CoarseLabels_to_FineLabels[CoarseLabel]:
        FineLabels_to_CoarseLabels[FineLabel] = CoarseLabel
        
# CREATE A DICTIONARY TO MAP THE INTEGER-ENCODED FINE LABEL TO THE WORD LABEL
# Create a list of the Fine Labels
FineLabels = list(FineLabels_to_CoarseLabels.keys())

# The target variable in CIFER100 is encoded such that the fine class is assigned an integer based on its alphabetical order
# Sort the fine class list.  
FineLabels.sort()
FineInts_to_FineLabels = dict(enumerate(FineLabels))


# CREATE A DICTIONARY TO MAP THE INTEGER-ENCODED FINE LABELS TO THE INTEGER-ENCODED COARSE LABELS
b = list(dict(sorted(FineLabels_to_CoarseLabels.items())).values())
FineInts_to_CoarseInts = dict(zip(range(100), [CoarseLabels_to_CoarseInts[i] for i in b]))


# CREATE A TENSORFLOW LOOKUP TABLE TO MAP THE INTEGER-ENCODED FINE LABELS TO THE INTEGER-ENCODED COARSE LABELS
table = StaticHashTable(
    initializer=KeyValueTensorInitializer(
        keys=list(FineInts_to_CoarseInts.keys()),
        values=list(FineInts_to_CoarseInts.values()),
        key_dtype=tf.int32, 
        value_dtype=tf.int32
    ),
    default_value=tf.constant(-1, tf.int32),
    name="dictionary"
)

# %%
# THIS CODE CELL IS TO BUILD A FUNCTIONAL MODEL

inputs = layers.Input(shape=input_shape)
x = layers.BatchNormalization()(inputs)

x = layers.Conv2D(64, (3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.30)(x)

x = layers.Conv2D(256, (3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.30)(x)

x = layers.Conv2D(256, (3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.30)(x)

x = layers.Conv2D(1024, (3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.30)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.30)(x)

x = layers.Dense(512, activation = "relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.30)(x)

output_fine = layers.Dense(num_fine_classes, activation="softmax", name="output_fine")(x)

model = Model(inputs=inputs, outputs=output_fine)

# %%
# THIS CODE CELL IS TO DEFINE A CUSTOM LOSS FUNCTION

# First, map the true fine labels to the true coarse labels
def get_y_true_coarse(y_true):
    y_true = tf.constant(y_true, dtype=tf.int32)
    y_true_coarse = table.lookup(y_true)
    return y_true_coarse

# Next, map the predicted fine class to the predicted coarse class (softmax probabilities)
initialize = tf.zeros(shape=(batch_size, num_coarse_classes), dtype=tf.float32)
y_pred_coarse = tf.Variable(initialize, dtype=tf.float32)
def get_y_pred_coarse(y_pred):
    for i in range(batch_size):
        for j in range(num_coarse_classes):
            idx = table.lookup(tf.range(100)) == j
            total = tf.reduce_sum(y_pred[i][idx])
            y_pred_coarse[i, j].assign(total)
    return y_pred_coarse

# Use the true coarse label and predicted coarse label (softmax probabilities) to derive the crossentropy loss of coarse labels
def hierarchical_loss(y_true, y_pred):
    y_true_coarse = get_y_true_coarse(y_true)
    y_pred_coarse = get_y_pred_coarse(y_pred)
    return SparseCategoricalCrossentropy()(y_true_coarse, y_pred_coarse)

# Use the true fine label and predicted finel label (softmax probabilities) to derive the crossentropy loss of fine labels
def crossentropy_loss(y_true, y_pred):
    return SparseCategoricalCrossentropy()(y_true, y_pred)

# Finally, combine the coarse class and fine class crossentropy losses 
def custom_loss(y_true, y_pred):
    H = 0.5
    total_loss = (1 - H) * crossentropy_loss(y_true, y_pred) + H * hierarchical_loss(y_true, y_pred)
    return total_loss

# %%
# THIS CODE CELL IS TO COMPILE THE MODEL

model.compile(optimizer="adam", loss=hierarchical_loss, metrics="accuracy", run_eagerly=True)

# %%
# THIS CODE CELL IS TO TRAIN THE MODEL

history = model.fit(x_train, y_train_fine_int, epochs=20, validation_split=0.25, batch_size=batch_size)

# %%
# THIS CODE CELL IS TO VISUALIZE THE TRAINING

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["accuracy", "val_accuracy"]].plot()
history_frame.loc[:, ["loss", "val_loss"]].plot()
plt.show()