import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []
			for p in self.preprocessors:
				image = p.preprocess(image)

			data.append(image)