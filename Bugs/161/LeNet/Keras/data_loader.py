import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class DataLoader():
	""" Load CIFAR dataset """
	def __init__(self, config):
		# Load config file
		self.config = config
		# Load data from data_path
		self.X_train, self.y_train, self.X_test, self.y_test = self.load_cifar10(config["data_path"])
		# Scale inputs 
		self.X_train, self.X_test = self.scale_data(self.X_train), self.scale_data(self.X_test)
		# Convert inputs to grayscale
		self.X_train, self.X_test = self.convert_grayscale(self.X_train), self.convert_grayscale(self.X_test)
		# One-hot encode the labels
		self.y_train, self.y_test = self.one_hot_labels(self.y_train), self.one_hot_labels(self.y_test)
		# Split train into train/validation
		self.X_train, self.X_valid, self.y_train, self.y_valid = self.split(self.X_train, self.y_train)

	def unpickle(self, file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict

	def load_cifar10(self, data_path):
		train_data = None
		train_labels = []
		test_data = None
		test_labels = None

		for i in range(1, 6):
			data_dict = self.unpickle(data_path + "data_batch_" + str(i))
			if (i == 1):
				train_data = data_dict[b'data']
			else:
				train_data = np.vstack((train_data, data_dict[b'data']))
			train_labels += data_dict[b'labels']

		test_data_dict = self.unpickle(data_path + "test_batch")
		test_data = test_data_dict[b'data']
		test_labels = test_data_dict[b'labels']

		train_data = train_data.reshape((50000, 3, 32, 32))
		train_data = np.rollaxis(train_data, 1, 4)
		train_labels = np.array(train_labels)

		test_data = test_data.reshape((10000, 3, 32, 32))
		test_data = np.rollaxis(test_data, 1, 4)
		test_labels = np.array(test_labels)

		return train_data, train_labels, test_data, test_labels		

	def scale_data(self, data):
		""" 
		Scale the row pixel intensities to the range [0, 1]
		"""

		data = data.astype(np.float32) / 255.0
		return data

	def convert_grayscale(self, data_image):
		"""
		Convert image to grayscale
		"""
		output = np.zeros((data_image.shape[:-1]))
		for i, image in enumerate(data_image):
			gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			output[i] = gray_image

		return output.reshape(-1, 32, 32, 1)

	def one_hot_labels(self, label_data, num_classes=10):
		"""
		One hot encode the labels
		"""
		label_data = np.eye(num_classes)[label_data.reshape(-1)]

		return label_data

	def split(self, train_data, train_labels):
		""" Split train_data into train/validation set """
		return train_test_split(train_data, train_labels, test_size=0.1, random_state=42)

	def next_batch(self):
		""" Yield batches for training """

		while True:
			idx = np.random.choice(self.X_train.shape[0], size=self.config["batch_size"])
			yield (self.X_train[idx], self.y_train[idx])

