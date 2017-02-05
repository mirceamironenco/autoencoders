import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 50
LEARNING_RATE = 1e-3
EPOCHS = 5

class ConvolutionalAutoencoder(object):
	def __init__(self):
		raise NotImplemented

	def inference(self, x, color_channels=1,
	              num_filters=(16, 16),
	              filter_sizes=(3, 3)):

		if len(num_filters) != len(filter_sizes):
			raise ValueError('Different number of filters specified for inference.')

		if len(x.get_shape()) not in [2, 4]:
			raise ValueError('Wrong input dimension. Should be (batch_size, vector_repr) '
			                 'or (batch_size, width, height, color_channels)')

		elif len(x.get_shape()) == 2:
			# Reshape to (batch_size, width, height, color_channels)
			square_dim = np.sqrt(x.get_shape().as_list()[1])
			layer_activation = tf.reshape(x, [-1, square_dim, square_dim, color_channels])
		else:
			# Dimension is (batch, w, h, d)
			layer_activation = x



		raise NotImplemented

	def loss(self, outputs, labels):
		raise NotImplemented

	def optimize(self, loss):
		raise NotImplemented

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

