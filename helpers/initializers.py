import tensorflow as tf

def he_xavier(in_size: int, out_size: int):
	"""
	Xavier initialization according to Kaiming He in:
	*Delving Deep into Rectifiers: Surpassing Human-Level
	Performance on ImageNet Classification
	(https://arxiv.org/abs/1502.01852)
	"""
	stddev = tf.cast(tf.sqrt(2 / in_size), tf.float32)
	W = tf.random_normal([in_size, out_size], stddev=stddev)
	b = tf.zeros([out_size])
	return tf.Variable(W, name="weights"), tf.Variable(b, name="biases")
