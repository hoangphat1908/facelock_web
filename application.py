from flask import Flask, request, json, render_template, redirect
import boto3
import botocore
import pickle
from encode_faces import add_encode_web
from detect_face import detect_web

BUCKET_NAME = 'facelock-sjsu'
USER_NAME = 'testuser'
IMAGES = 'images'
MODEL_FILE_NAME = 'encodings.pickle'
CASCADE = 'haarcascade_frontalface_default.xml'
URL = 'https://s3.amazonaws.com/' + BUCKET_NAME

application = Flask(__name__)
S3 = boto3.resource('s3')

@application.route('/', methods = ['GET'])
def index():
    images = get_images()
    return render_template('index.html', images=images)

@application.route('/', methods = ['POST'])
def encode():
    f = request.files.getlist('file')
    method = 'hog'
    name = request.form.get('name')
    
    model = load_model(USER_NAME + '/' + MODEL_FILE_NAME)

    image, new_model = add_encode_web(f, model, method, name)
    write_model(new_model, USER_NAME + '/' + MODEL_FILE_NAME)
    write_image(image, USER_NAME + '/' + IMAGES + '/' + name + '.jpg')
    return redirect('/')

@application.route('/predict', methods=['POST'])
def predict():
    image = request.files['image'].read()
    model = load_model(USER_NAME + '/' + MODEL_FILE_NAME)
    prediction = detect_web(CASCADE, model, image)
    result = {'prediction': prediction}    
   
    return json.dumps(result)

def load_model(key):   
    model_str = None
    try:
        S3.Object(BUCKET_NAME, key).load()
    except botocore.exceptions.ClientError as e:
        return model_str
    else:
        response = S3.Object(BUCKET_NAME, key)
        model_str = response.get()['Body'].read()
    return model_str

def write_model(file, key):     
    S3.Object(BUCKET_NAME, key).put(Body=file)

def write_image(file, key):     
    S3.Bucket(BUCKET_NAME).put_object(Key=key, Body=file, ACL='public-read')

def get_images():
    images=[]
    bucket = S3.Bucket(BUCKET_NAME)
    for image in bucket.objects.filter(Prefix=USER_NAME + '/images'):
        image_info= [get_user(image.key), URL + '/' + image.key]
        images.append(image_info)
    return images

def get_user(key):
    start = key.find('images/') + 7
    end = key.find('.jpg', start)
    return key[start:end]

if __name__ == "__main__":
    application.debug = True
    application.run()