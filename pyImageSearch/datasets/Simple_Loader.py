import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbos=-1):
		# initailize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for i, imagePath in enumerate(imagePaths):
			# load the image and extract the class label
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessor is not None:
				# loop over the preprocessors and apply each to the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a feature vector
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every verbose image
			if verbos > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))