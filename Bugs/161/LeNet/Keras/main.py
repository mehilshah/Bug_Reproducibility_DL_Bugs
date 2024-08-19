import json
from data_loader import DataLoader
import model
from trainer import Trainer 

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load Cifar data 
	data = DataLoader(config)

	# Create LeNet model
	LeNet = model.LeNet(config)

	# Create trainer
	trainer = Trainer(LeNet.model, data, config)

	# Train model
	trainer.train()

	# # Save LeNet model weights
	# LeNet.save_weights()

	# # Load weights
	# LeNet.load_weights(config["model_load_weights_path"])

	# Evaluate validation set
	trainer.evaluate()

if __name__ == '__main__':
	main()