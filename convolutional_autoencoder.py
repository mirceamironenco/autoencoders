import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from helpers.utils import print_flags

BATCH_SIZE = 50
LEARNING_RATE = 1e-2
EPOCHS = 10


class ConvolutionalAutoencoder(object):
	def __init__(self, learning_rate=1e-2):
		self.learning_rate = learning_rate

	def inference(self, x, color_channels=1,
	              output_sizes=(16, 16, 16),
	              filter_sizes=(3, 3, 3)):

		if len(output_sizes) != len(filter_sizes):
			raise ValueError('Different number of filters specified for inference.')

		if len(x.get_shape()) not in [2, 4]:
			raise ValueError('Wrong input dimension. Should be (batch_size, vector_repr) '
			                 'or (batch_size, width, height, color_channels)')

		elif len(x.get_shape()) == 2:
			# Reshape to (batch_size, width, height, color_channels)
			square_dim = int(np.sqrt(x.get_shape().as_list()[1]))
			x_reshaped = tf.reshape(x, [-1, square_dim, square_dim, color_channels])
		else:
			# Dimension is (batch, w, h, d)
			x_reshaped = x

		layer_activation = x_reshaped

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
			convolution = tf.nn.bias_add(tf.nn.conv2d(layer_activation, filter=W,
			                                       strides=[1, 2, 2, 1], padding='SAME'),
			                          b)
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
			deconvolution = tf.nn.bias_add(tf.nn.conv2d_transpose(layer_activation, W,
			                                                   tf.pack(
				                                                   [tf.shape(x)[0], s[1],
				                                                    s[2], s[3]]),
			                                                   strides=[1, 2, 2, 1],
			                                                   padding='SAME'), b)
			layer_activation = tf.nn.elu(deconvolution)

		output = layer_activation
		return output, x_reshaped

	def loss(self, outputs, labels):
		return tf.reduce_sum(tf.square(outputs - labels))

	def optimize(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		return optimizer.minimize(loss)

def main(_):
	print_flags(FLAGS)

	input_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
	model = ConvolutionalAutoencoder(learning_rate=FLAGS.learning_rate)
	output, label = model.inference(x=input_placeholder)
	loss = model.loss(output, label)
	optimization_op = model.optimize(loss)

	# Load MNIST data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	X_train = mnist.train.images
	mean_image = np.mean(X_train, axis=0)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for epoch in range(FLAGS.epochs):
		for batch_num in range(mnist.train.num_examples // FLAGS.batch_size):
			batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
			batch_x = np.array([img - mean_image for img in batch_x])
			_ = sess.run(optimization_op, feed_dict={input_placeholder: batch_x})
		print('Epoch {} Loss {}'.format(epoch + 1, sess.run(loss,
		                                                    feed_dict={
			                                                    input_placeholder: batch_x})))

	n = 10
	x_test, _ = mnist.test.next_batch(n)
	x_test = np.array([img - mean_image for img in x_test])
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
		plt.imshow(np.reshape(np.reshape(recon_imgs[i], (784,)) + mean_image, (28,28)))
		# plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

FLAGS = None
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
	parser.add_argument('--epochs', type=int, default=EPOCHS)
	FLAGS, _ = parser.parse_known_args()

	tf.app.run()
