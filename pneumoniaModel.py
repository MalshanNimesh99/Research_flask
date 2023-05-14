import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)

model =load_model('Pneumonia10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Not a Pneumonia patient"
	elif classNo==1:
		return "A Pneumonia patient"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result


@app.route('/predictPneumonia', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']
        basepath = os.path.dirname(__file__)

        file_path = os.path.join(basepath, 'uploadsPn', secure_filename(f.filename))
        f.save(file_path)

        value=getResult(file_path)
        print(value)

        result=get_className(value)
        print('result : ' + result)

        response = {'result' : result}
        return jsonify(response)
    
    return None


if __name__ == '__main__':
    app.run(debug=True)