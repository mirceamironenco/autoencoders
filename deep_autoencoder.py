import argparse
import tensorflow as tf
import numpy as np

BATCH_SIZE = 256
LEARNING_RATE = 1e-5

class DeepAutoencoder(object):
	def __init__(self):
		raise NotImplemented

	def inference(self, x):
		raise NotImplemented

	def loss(self, logits, labels):
		raise NotImplemented

	def optimize(self, loss):
		raise NotImplemented

def train():
	return 0

def main(_):
	print_flags()
	train()

def print_flags():
	for key, value in vars(FLAGS).items():
		print("{}: {}".format(key, str(value)))

FLAGS = None
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
	FLAGS, _ = parser.parse_known_args()

	tf.app.run()

