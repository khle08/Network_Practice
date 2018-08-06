from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing import Simple_Preprocessor
from datasets import Simple_Loader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images... ")
imagePaths = list(paths.list_images(args["dataset"]))

''' initialize the image preprocessor, load the dataset from disk,
and reshape the data matrix'''
sp = Simple_Preprocessor(32, 32)
sdl = Simple_Loader(preprocessors=[sp])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

''' partition the data into training and testing splits using 75% of
the data for training and the remaining 25% for testing'''
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

for r in (None, "l1", "l2"):
	print("[INFO] training model with '{}' panelty".format(r))
	model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=42)
	model.fit(trainX, trainY)

	acc = model.score(testX, testY)
	print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc * 100))

