import numpy as np

class Perceptron:
	def __init__(self, N, alpha=0.1):
		self.W = np.random.randn(N+1) / np.sqrt(N)
		self.alpha = alpha

	def step(self, x):
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs=10):
		''' insert a column of 1's as the last entry 
		in the feature matrix -- this little trick allows
		us to treat the bias as a trainable parameter 
		within the weight matrix'''
		X = np.c_[X, np.ones((X.shape[0]))]

		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
				p = self.step(np.dot(x, self.W))

				if p != target:
					error = p - target

					self.W += -self.alpha * error * x

		