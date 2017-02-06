import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from helpers.initializers import he_xavier

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10

class VariationalAutoencoder(object):
	"""
	Kingma & Welling - Auto-Encoding Variational Bayes
	(http://arxiv.org/abs/1312.6114)
	"""
	def __init__(self, learning_rate=0.001, batch_size=128,
	             architecture=(),
	             non_linear=tf.nn.elu,
	             squashing=tf.nn.sigmoid):
		"""
		Initialize VAE model setting specified parameters.

		Args:
			- architecture. Encoder+Decoder architecture, including
			input dimension as first value in the array. Example:
			[728, 512, 256, 64] is a VAE with 728-sized inputs and a
			latent space of size 64. The complete architecture will be:
			[728, 512, 256, 64, 256, 512, 728].
			- non_linear. Non-linear activation functions used by the networks.
		"""
		self.learning_rate = learning_rate
		self.architecture = architecture
		self.batch_size = batch_size
		self.non_linear = non_linear
		self.squashing = squashing
		self.latent_size = architecture[-1]

	def construct_encoding(self, x):
		input = x

		for dim_i, layer_size in enumerate(self.architecture[1:-1]):
			# Weights init
			W, b = he_xavier(in_size=self.architecture[dim_i], out_size=layer_size)

			# Layer activation
			layer_activ = self.non_linear(tf.add(tf.matmul(input, W), b))
			input = layer_activ

		return input, layer_size

	def construct_decoding(self, z):
		input = z
		in_size = self.latent_size
		for layer_size in self.architecture[::-1][1:-1]:
			W, b = he_xavier(in_size=in_size, out_size=layer_size)
			layer_activ = self.non_linear(tf.add(tf.matmul(input, W), b))
			input, in_size = layer_activ, layer_size

		return input, layer_size

	def inference(self, x):
		# Compute encoding / recognition q(z|x)
		h_encoded, h_size = self.construct_encoding(x)

		# Compute latent distribution parameters
		# z ~ N(z_mean, exp(z_log_sigma)^2)
		W_mean, b_mean = he_xavier(in_size=h_size,
		                 out_size=self.latent_size)
		W_sigma, b_sigma = he_xavier(in_size=h_size,
		                             out_size=self.latent_size)

		z_mean = tf.add(tf.matmul(h_encoded, W_mean), b_mean)
		z_log_sigma = tf.add(tf.matmul(h_encoded, W_sigma), b_sigma)

		# Sample from the latent distribution to compute reconstruction
		z = VariationalAutoencoder.sample_normal(mu=z_mean, log_sigma=z_log_sigma)

		# Final decoding layer
		h_final, final_size = self.construct_decoding(z)

		# Compute decoding / reconstruction p(x|z)
		W_x, b_x = he_xavier(in_size=final_size, out_size=self.architecture[0])
		x_reconstructed = self.squashing(tf.add(tf.matmul(h_final, W_x), b_x))

		return x_reconstructed

	def loss(self, outputs, labels):
		raise NotImplemented

	def optimize(self, loss):
		raise NotImplemented

	@staticmethod
	def sample_normal(mu, log_sigma):
		"""
		Re-parametrization trick
		z = z_mean + exp(z_log_sigma) * epsilon
		"""
		epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
		return mu + tf.exp(log_sigma) * epsilon

def main(_):

	# Load MNIST data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	input = tf.placeholder(tf.float32, shape=[None, 784])

	model = VariationalAutoencoder(learning_rate=FLAGS.learning_rate,
	                               batch_size=FLAGS.batch_size,
	                               architecture=[784, 512, 512, 2])

	reconstruction = model.inference(input)

FLAGS = None
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
	parser.add_argument('--epochs', type=int, default=EPOCHS)
	FLAGS, _ = parser.parse_known_args()

	tf.app.run()
