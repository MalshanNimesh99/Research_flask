from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size = 256

app = Flask(__name__)
CORS(app)

model = load_model('mriDementia.hdf5')

label_dict = {
      0: 'Mild Demented',
      1: 'Moderate Demented',
      2: 'Non Demented',
      3: 'Very Mild Demented'
    }

def preprocess(img):

    img = np.array(img)

    # if(img.ndim==3):
    #     gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray=img

    # gray=gray/255
    resized=cv2.resize(img, (img_size, img_size))
    reshaped = resized.reshape(1,img_size, img_size)
    return reshaped

# @app.route("/")
# def index():
#     return(render_template("index.html"))

@app.route("/predict", methods = ["POST"])
def predict():
    print('HERE')
    
    # img = Image.open(request.files['file'].stream)
    

    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    dataBytesIO.seek(0)
    image = Image.open(dataBytesIO)

    test_image = preprocess(image)
    image = np.repeat(np.expand_dims(test_image, axis=-1), 3, axis=-1)

    prediction = model.predict(image)
    result = np.argmax(prediction, axis=1)[0]
    accuracy = float(np.max(prediction, axis=1)[0])

    label = label_dict[result]

    print(prediction, result, accuracy)

    response = {'Prediction' : {'result' : label, 'accuracy' : accuracy}}

    return jsonify(response)

app.run(debug = True)