import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 50
LEARNING_RATE = 1e-3
EPOCHS = 5

class ConvolutionalAutoencoder(object):
	def __init__(self, learning_rate=1e-3):
		self.learning_rate = learning_rate

	def inference(self, x, color_channels=1,
				  output_sizes=(16, 16),
				  filter_sizes=(3, 3)):

		if len(output_sizes) != len(filter_sizes):
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

		# Construct encoder
		input_depth = color_channels

		# Store weights to be reused
		encoder = []

		# Store output shapes to be used for
		# deconvolution operation, output_shape parameter.
		output_shapes = []
		for filter_index, filter_size in enumerate(filter_sizes):
			# Current output activation size (# of filters).
			output_size = output_sizes[filter_index]

			# Initialize weights
			W = tf.Variable(tf.random_uniform([filter_size, filter_size,
											   input_depth, output_size],
											  minval=-1.0 / np.sqrt(input_depth),
											  maxval=1.0 / np.sqrt(input_depth)))
			b = tf.Variable(tf.zeros([output_size]))

			# Save weights to be used by decoder (tied weights)
			encoder.append(W)
			output_shapes.append(layer_activation.get_shape().as_list())

			# Get layer output activation
			convolution = tf.bias_add(tf.nn.conv2d(layer_activation, filter=W,
											  strides=[1, 2, 2, 1], padding='SAME'), b)
			layer_activation = tf.nn.elu(convolution)

			input_depth = layer_activation.get_shape().as_list()[3]

		encoder.reverse()
		output_shapes.reverse()

		# Construct decoder
		# Note: tf.nn.conv2d_transpose is sometimes called 'Deconvolution', but it is
		# transpose(gradient) of conv2d.
		for weight_index, weight in enumerate(encoder):
			W = weight
			b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
			s = output_shapes[weight_index]
			deconvolution = tf.bias_add(tf.nn.conv2d_transpose(layer_activation, W,
			                                                   tf.pack([tf.shape(x)[0], s[1],
			                                                           s[2], s[3]]),
			                                                   strides=[1, 2, 2, 1],
			                                                   padding='SAME'), b)
			layer_activation = tf.nn.elu(deconvolution)

		output = layer_activation
		raise output

	def loss(self, outputs, labels):
		return tf.reduce_sum(tf.square(outputs - labels))

	def optimize(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		return optimizer.minimize(loss)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

