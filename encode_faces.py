# USAGE
# When encoding on laptop, desktop, or GPU (slower, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method cnn
# When encoding on Raspberry Pi (faster, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import numpy as np

def encode(dataset, encodings_path, method):
	# grab the paths to the input images in our dataset
	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(dataset))

	# initialize the list of known encodings and known names
	knownEncodings = []
	knownNames = []

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb,
			model=method)

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		# loop over the encodings
		for encoding in encodings:
			# add each encoding + name to our set of known names and
			# encodings
			knownEncodings.append(encoding)
			knownNames.append(name)

	# dump the facial encodings + names to disk
	print("[INFO] serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames}
	f = open(encodings_path, "wb")
	f.write(pickle.dumps(data, protocol=0))
	f.close()

def add_encode_web(image_files, encodings_path, method, name):
	# grab the paths to the input images in our dataset
	print("[INFO] quantifying faces...")
	#imagePaths = list(paths.list_images(dataset))
	knownEncodings = []
	knownNames = []
	# initialize the list of known encodings and known names
	
	if encodings_path is not None:
		data = pickle.loads(encodings_path, encoding='latin1')
		knownEncodings = data["encodings"]
		knownNames = data["names"]

	i = 0
	return_image = None
	for f in image_files:
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(image_files)))

		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)

		image_string = f.read()
		nparr = np.fromstring(image_string, np.uint8)
		image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb,
			model=method)

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		# loop over the encodings
		for encoding in encodings:
			# add each encoding + name to our set of known names and
			# encodings
			knownEncodings.append(encoding)
			knownNames.append(name)

		if i == 0:
			return_image = image_string
		i += 1

	# dump the facial encodings + names to disk
	print("[INFO] serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames}
	return return_image, pickle.dumps(data, protocol=0)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--dataset", required=True,
		help="path to input directory of faces + images")
	ap.add_argument("-e", "--encodings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-d", "--detection-method", type=str, default="cnn",
		help="face detection model to use: either `hog` or `cnn`")
	args = vars(ap.parse_args())

	encode(args["dataset"], args["encodings"], args["detection_method"])