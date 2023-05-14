import os
import tensorflow as tf
import numpy as np
from flask_cors import CORS
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)
cors = CORS(app)

model =load_model('BrainTumorApril.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "There is not a Brain Tumor"
	elif classNo==1:
		return "There is a Brain Tumor"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result

@app.route('/predictBrainTumor', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']
        print(f)

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploadsBt', secure_filename(f.filename))
        f.save(file_path)

        value=getResult(file_path)
        print(value)

        result=get_className(value)
        print('result is : ' + result)

        response = {'result' : result}
        return jsonify(response)
        
    return None

if __name__ == '__main__':
    app.run(debug=True)