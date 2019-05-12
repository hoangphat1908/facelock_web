from flask import Flask, request, json, render_template
import boto3
import pickle
from facelock import detect

BUCKET_NAME = 'facelock-sjsu'
USER_NAME = 'testuser'
MODEL_FILE_NAME = 'encodings.pickle'
CASCADE = 'haarcascade_frontalface_default.xml'

# EB looks for an 'application' callable by default.
application = Flask(__name__)
S3 = boto3.client('s3')

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    image = request.files['image'].read()
    model = load_model(USER_NAME + '/' + MODEL_FILE_NAME)
    prediction = detect(CASCADE, model, image)
    result = {'prediction': prediction}    
   
    return json.dumps(result)

def load_model(key):    
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()     
    return model_str

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()