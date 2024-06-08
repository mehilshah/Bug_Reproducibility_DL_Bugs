from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random 
batch_size = random.randint(1, 100)
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_dir = './PetImages/train'
test_dir = './PetImages/test'
validation_dir = './PetImages/validation'

train_data_gen = train_image_generator.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
val_data_gen =validation_image_generator.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        shuffle = False,)