from sklearn.neighbors import KneighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import Simple_Preprocessor
from pyimagesearch.datasets import Simple_Loader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, 
	help='path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1, 
	help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1, 
	help='#of jobs for k-NN distance(-1 uses all available cores)')
args = vars(ap.parse_args())

