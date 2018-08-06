# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

def predict(X, W):
	preds = sigmoid_activation(X.dot(W))
	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1
	return preds

def next_batch(X, y, batchSize):
	''' loop over our dataset 'X' in nmini-batchs, yielding
	a tuple of the current batched data and labels'''
	for i in np.arange(0, X.shape[0], batchSize):
		yield(X[i:i + batchSize], y[i:i + batchSize])


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="size of SGD mini-batches")
args = vars(ap.parse_args())

''' generate a 2-class 