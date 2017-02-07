from datetime import datetime
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from helpers.initializers import he_xavier
from helpers.utils import print_flags, load_mnist

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 75

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
		assert len(architecture) > 1, "Please specify at least input and latent size."
		self.x_in = tf.placeholder(tf.float32, shape=[None, architecture[0]])
		self.learning_rate = learning_rate
		self.architecture = architecture
		self.batch_size = batch_size
		self.non_linear = non_linear
		self.squashing = squashing
		self.z_size = architecture[-1]
		self._sess = tf.Session()
		self.decoder_vars = {}

		# Construct graph operations and keep reference to nodes.
		self.x_rec, self.z_mu, self.z_log_sigma, self.x_gen, self.z_gen = self.inference()
		self.vae_loss = self.loss(input_reconst=self.x_rec,
		                          z_mu=self.z_mu,
		                          z_log_sigma=self.z_log_sigma)
		self.train_op = self.optimize(self.vae_loss)

		# Initialize session to run operations in.
		self._sess.run(tf.initialize_all_variables())

	def construct_encoding(self):
		input = self.x_in

		for dim_i, layer_size in enumerate(self.architecture[1:-1]):
			# Weights init
			W, b = he_xavier(in_size=self.architecture[dim_i], out_size=layer_size)

			# Layer activation
			layer_activ = self.non_linear(tf.add(tf.matmul(input, W), b))
			input = layer_activ

		return input, layer_size

	def construct_decoding(self, z, reuse=False):
		input = z
		in_size = self.z_size
		for dim_i, layer_size in enumerate(self.architecture[::-1][1:-1]):
			if reuse: # TO-DO: Change to tf variable reuse instead of dict.
				W, b = self.decoder_vars[dim_i]
			else:
				W, b = he_xavier(in_size=in_size, out_size=layer_size)
				self.decoder_vars[dim_i] = (W,b)
			layer_activ = self.non_linear(tf.add(tf.matmul(input, W), b))
			input, in_size = layer_activ, layer_size

		return input, layer_size

	def inference(self):
		# Compute encoding / recognition q(z|x)
		h_encoded, h_size = self.construct_encoding()

		# Compute latent distribution parameters
		# z ~ N(z_mean, exp(z_log_sigma)^2)
		W_mean, b_mean = he_xavier(in_size=h_size,
		                           out_size=self.z_size)
		W_sigma, b_sigma = he_xavier(in_size=h_size,
		                             out_size=self.z_size)

		z_mu = tf.add(tf.matmul(h_encoded, W_mean), b_mean)
		z_log_sigma = tf.add(tf.matmul(h_encoded, W_sigma), b_sigma)

		# Sample from the latent distribution to compute reconstruction
		z = self.sample_normal(mu=z_mu, log_sigma=z_log_sigma)

		# Final decoding layer
		h_final, final_size = self.construct_decoding(z)

		# Compute decoding / reconstruction p(x|z)
		W_x, b_x = he_xavier(in_size=final_size, out_size=self.architecture[0])
		x_reconstructed = self.squashing(tf.add(tf.matmul(h_final, W_x), b_x))

		# Generating new points from latent space operation
		# Define z sample to be used for generation, default from prior z ~ N(0,I)
		z_gen = tf.placeholder_with_default(tf.random_normal([1, self.z_size]),
		                                    shape=[None, self.z_size])
		h, f = self.construct_decoding(z_gen, reuse=True)
		x_gen = self.squashing(tf.add(tf.matmul(h, W_x), b_x))  # Not used for training.

		return x_reconstructed, z_mu, z_log_sigma, x_gen, z_gen

	def loss(self, input_reconst, z_mu, z_log_sigma):
		"""
		Compute total loss
		"""
		reconst_loss = VariationalAutoencoder.cross_entropy(input_reconst,
		                                                           self.x_in)
		divergence_loss = VariationalAutoencoder.kullback_leibler(mu=z_mu,
		                                                          log_sigma=z_log_sigma)

		return tf.reduce_mean(reconst_loss + divergence_loss, name="vae_loss")

	def optimize(self, loss):
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		tvars = tf.trainable_variables()
		grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=tvars)
		clipped = [(tf.clip_by_value(gradient, -5, 5), tvar)
		           for gradient, tvar in grads_and_vars]
		optimize_op = optimizer.apply_gradients(clipped, name="cost_minimization")
		return optimize_op

	def train_step(self, x):
		_, loss = self._sess.run([self.train_op, self.vae_loss], feed_dict={self.x_in: x})
		return loss

	def encode_input(self, x):
		return self._sess.run([self.z_mu, self.z_log_sigma], feed_dict={self.x_in: x})

	def decode_input(self, z_sample=None):
		# if z_sample=None, we simply use default from prior z ~ N(0,I)
		# if z_sample is tensor, first obtain numpy value, otherwise use it directly.
		if z_sample is not None:
			z_sample = self._sess.run(z_sample) if hasattr(z_sample, "eval") else z_sample

		return self._sess.run(self.x_gen, feed_dict={self.z_gen: z_sample})

	def variational_ae(self, x):
		z = self.sample_normal(*self.encode_input(x))
		return self.decode_input(z_sample=z)

	def sample_normal(self, mu, log_sigma):
		"""
		Re-parametrization trick
		z = z_mean + exp(z_log_sigma) * epsilon
		"""
		epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
		return mu + tf.exp(log_sigma) * epsilon

	@staticmethod
	def cross_entropy(res, label, offset=1e-7):
		res_ = tf.clip_by_value(res, offset, 1-offset)
		return -tf.reduce_sum(label * tf.log(res_) + (1 - label) * tf.log(1 - res_), 1)

	@staticmethod
	def kullback_leibler(mu, log_sigma):
		"""
		Kullback-Leibler divergence KL(q||p)
		"""
		return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - tf.square(mu)
		                            - tf.exp(2 * log_sigma), 1)


def main(_):
	print_flags(FLAGS)
	mnist = load_mnist()
	vae = VariationalAutoencoder(learning_rate=FLAGS.learning_rate,
	                               batch_size=FLAGS.batch_size,
	                               architecture=[784, 512, 512, 2])

	print("Started training {}".format(datetime.now().isoformat()[11:]))
	# Run training
	for epoch in range(FLAGS.epochs):
		for batch in range(mnist.train.num_examples // FLAGS.batch_size):
			batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
			_ = vae.train_step(x=batch_x)
		print("Epoch {} Loss {}".format(epoch + 1, vae.train_step(x=batch_x)))
	print("Training finished, {}.".format(datetime.now().isoformat()[11:]))

	x_sample = mnist.test.next_batch(100)[0]
	x_rec = vae.variational_ae(x_sample)

	plt.figure(figsize=(8, 12))
	for i in range(5):
		plt.subplot(5, 2, 2 * i + 1)
		plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
		plt.title("Test input")
		plt.colorbar()
		plt.subplot(5, 2, 2 * i + 2)
		plt.imshow(x_rec[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
		plt.title("Reconstruction")
		plt.colorbar()
	plt.tight_layout()
	plt.savefig("vae_digits.png")

	x_sample, y_sample = mnist.test.next_batch(5000)
	z_mu, _ = vae.encode_input(x_sample)
	plt.figure(figsize=(8, 6))
	plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
	plt.colorbar()
	plt.grid()
	plt.savefig("vae_latent.png")


FLAGS = None
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
	parser.add_argument('--epochs', type=int, default=EPOCHS)
	FLAGS, _ = parser.parse_known_args()

	tf.app.run()
