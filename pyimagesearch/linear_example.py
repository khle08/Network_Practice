import numpy as np
import cv2

# to initialize the class labels and set the seed of the pseudorandom number generator
labels = ["dog", "cat", "panda"]
np.random.seed(1)


W = np.random.randn(3, 3072)
b = np.random.randn(3)

# to load the example image, resize it, and flatten it into "feature vector"
orig = cv2.imread("/home/kcl/Desktop/img_2.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

# compute the output scores by taking the dot product between the weight matrix and image pixels
scores = W.dot(image) + b

for label, score in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the heighest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our input image
cv2.imshow("image", orig)
cv2.waitKey(0)