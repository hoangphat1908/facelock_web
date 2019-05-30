# Requirements (requirements.txt):
- opencv-python
- dlib
- face-recognition
- imutils
- flask
- numpy
# Running instructions
- Configure the AWS CLI with an IAM user 
- Change user and bucket name in application.py file
- Create a virtual environment (optional): 
  + virtualenv env 
  + source env/bin/activate
- Install dependencies: 
  + pip install -r requirements.txt
- Run the flask application:
  + python application.py
<a href="https://imgur.com/IO6VVR9"><img src="https://i.imgur.com/IO6VVR9.png?1" title="source: imgur.com" /></a>
- Add a lock owner with some images
- Detect a single image 
  + HTTP POST request to localhost:5000/predict
  + Include an image file with key “image”

# Encode / Detect faces locally (scripts.txt)
- Encode all faces in the dataset folder with hog method, into encodings.pickle file:
  + python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog
- Detect a single image using the pickle file and the Haar Cascade method
  + python detect_face.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle --image test_images/phat.jpg
