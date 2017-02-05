import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 50
LEARNING_RATE = 1e-3
EPOCHS = 5

class VariationalAutoencoder(object):
	def __init__(self):
		raise NotImplemented

	def inference(self, x):
		raise NotImplemented

	def loss(self, outputs, labels):
		raise NotImplemented

	def optimize(self, loss):
		raise NotImplemented

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

