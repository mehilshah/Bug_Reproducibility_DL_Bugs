from keras.callbacks import ModelCheckpoint, TensorBoard

class Trainer(object):
	def __init__(self, model, data, config):
		self.model = model
		self.data = data 
		self.config = config
		self.callbacks = []

		self.init_callbacks()

	def init_callbacks(self):
		"""
		Create callbacks 
		"""

		self.callbacks.append(
			ModelCheckpoint(self.config["model_checkpoint_path"], 
				monitor="val_acc",
				save_best_only=True,
				mode="max", 
				verbose=0,
				)
			)

		self.callbacks.append(
			TensorBoard(
				log_dir="./logs/" + self.config["experiment_name"], 
				histogram_freq=1, 
				batch_size=self.config["batch_size"],
				write_graph=True, 
				)
			)


	def train(self):
		"""
		Train the model
		"""

		self.model.fit_generator(generator=(self.data.next_batch()),
			validation_data=(self.data.X_valid, self.data.y_valid),  
			epochs=self.config["num_epochs"],
			steps_per_epoch=self.config["num_iter_per_epoch"], 
			callbacks=self.callbacks,
			verbose=1
			)


	def evaluate(self):
		"""
		Evaluate test set
		"""
		results = self.model.evaluate(
			x=self.data.X_test, 
			y=self.data.y_test, 
			batch_size=self.config["batch_size"], 
			verbose=0
			)

		print("test loss = ", results[0])
		print("test accuracy = %.4f %%" % (results[1] * 100))

		# ///////// Added By @Mehdi
		import json
		import numpy as np
		file = open(file="result.json", mode="w")  
		model_acc = np.float64(results[1])
		res = {"val_acc" : model_acc}
		json.dump(res, file)
		file.close()

