import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5

class DeepAutoencoder(object):
	def __init__(self, learning_rate=1e-3):
		self.learning_rate = learning_rate

	def inference(self, x, dimensions=(512, 256, 128)):
		# Set network dimensions
		input_shape = int(x.get_shape()[1])
		dimensions = [input_shape] + list(dimensions)

		layer_activation = x

		# Construct encoder
		encoder_weights = []
		for output_dim in dimensions:
			W = tf.Variable(tf.random_uniform([input_shape, output_dim],
			                                  minval=-1.0 / np.sqrt(input_shape),
			                                  maxval=1.0 / np.sqrt(input_shape)))
			b = tf.Variable(tf.zeros([output_dim]))
			# Save weight to be reused by decoder
			encoder_weights.append(W)

			# Get latest layer activation
			layer_activation = tf.nn.tanh(tf.matmul(layer_activation, W) + b)

			# Set new input dimension
			input_shape = output_dim

		# Construct decoder with tied weights
		encoder_weights.reverse()

		for curr_index, output_dim in enumerate(dimensions[:-1][::-1]):
			W = tf.transpose(encoder_weights[curr_index])
			b = tf.Variable(tf.zeros(output_dim))
			layer_activation = tf.nn.tanh(tf.matmul(layer_activation, W) + b)

		output = layer_activation

		return output

	def loss(self, outputs, labels):
		return tf.reduce_sum(tf.square(outputs - labels))

	def optimize(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		return optimizer.minimize(loss)

def main(_):
	print_flags()

	# MNIST input placeholder
	input_placeholder = tf.placeholder(tf.float32, shape=[None, 784])

	# Construct model
	model = DeepAutoencoder(learning_rate=FLAGS.learning_rate)

	# Inference output, get optimization op.
	output = model.inference(input_placeholder, dimensions=(256, 64))
	loss = model.loss(output, input_placeholder)
	optimization_op = model.optimize(loss)

	# Load MNIST data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	X_train = mnist.train.images
	mean_image = np.mean(X_train, axis=0)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for epoch in range(FLAGS.epochs):
		for batch_number in range(mnist.train.num_examples // FLAGS.batch_size):
			# Get next batch of data and normalize
			batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
			# batch_x = np.array([img - mean_image for img in batch_x])
			batch_x = batch_x - mean_image
			_ = sess.run(optimization_op,
			                            feed_dict={input_placeholder: batch_x})
		# Print loss on last batch
		print('Epoch {} Loss {}'.format(epoch + 1, sess.run(loss,
		                                                feed_dict={input_placeholder: batch_x})))

	n = 10
	x_test, _ = mnist.test.next_batch(n)
	x_test -= mean_image
	recon_imgs = sess.run(output, feed_dict={input_placeholder: x_test})

	plt.figure(figsize=(20, 4))
	for i in range(n):
		# Display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_test[i].reshape(28, 28))
		# plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# Display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(np.reshape([recon_imgs[i] + mean_image], (28, 28)))
		# plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def print_flags():
	for key, value in vars(FLAGS).items():
		print("{}: {}".format(key, str(value)))

FLAGS = None
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
	parser.add_argument('--epochs', type=int, default=EPOCHS)
	FLAGS, _ = parser.parse_known_args()

	tf.app.run()
