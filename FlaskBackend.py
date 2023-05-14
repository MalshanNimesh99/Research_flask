from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 


app = Flask(__name__)
CORS(app)


model = load_model('mriDementia.hdf5')
model2 = load_model('BrainTumorApril.h5')
model3 =load_model('Pneumonia10Epochs.h5')


label_dict = {
      0: 'Mild Demented',
      1: 'Moderate Demented',
      2: 'Non Demented',
      3: 'Very Mild Demented'
    }

def preprocess(img):
    img = np.array(img)
    resized=cv2.resize(img, (256, 256))
    reshaped = resized.reshape(1, 256, 256)
    return reshaped

def get_className1(classNo):
    if classNo==0:
        return "No Brain Tumor"
    elif classNo==1:
        return "Brain Tumor Detected"

def getResult1(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model2.predict(input_img)
    return result

def get_className(classNo):
    if classNo==0:
        return "Not a Pneumonia Patient"
    elif classNo==1:
        return "Pneumonia Patient Detected"

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model3.predict(input_img)
    return result

@app.route("/predict", methods = ["POST"])
def predict():
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

@app.route('/predictBrainTumor', methods=['GET', 'POST'])
def predict_brain_tumor():
    if request.method == 'POST':
        f = request.files['file']
        print(f)
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploadsBt', secure_filename(f.filename))
        f.save(file_path)
        value=getResult1(file_path)
        print(value)
        result=get_className1(value)
        print('result is : ' + result)
        response = {'result' : result}
        return jsonify(response)
    return None

@app.route('/ckd', methods=['POST'])
def predict_ckd():
    if request.method == 'POST':
        req_data = request.get_json()
        k_Symptom_1 = float(req_data['Symptom_1'])
        k_Symptom_2 = float(req_data['Symptom_2'])
        k_Symptom_3 = float(req_data['Symptom_3'])
        k_Symptom_4 = float(req_data['Symptom_4'])
        k_Symptom_5 = float(req_data['Symptom_5'])
        k_Symptom_6 = float(req_data['Symptom_6'])
        k_Symptom_7 = float(req_data['Symptom_7'])
        k_Symptom_8 = float(req_data['Symptom_8'])

        data = pd.read_csv("new.csv")
        pd.set_option('display.max_columns', None) # will show the all columns with pandas dataframe
        pd.set_option('display.max_rows', None) # will show the all rows with pandas dataframe

        data['M/F'] = [1 if each == "M" else 0 for each in data['M/F']]
        data['Group'] = [1 if each == "Demented" or each == "Converted" else 0 for each in data['Group']]

        correlation_matrix = data.corr()
        data_corr = correlation_matrix['CDR'].sort_values(ascending=False)
        # print(data_corr)

        y = data['CDR'].values
        X = data[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size= 0.20, random_state=42, stratify=y)


        with open('CDR_model.pkl','rb') as file:
              mp = pickle.load(file)

        with open('model.pkl','rb') as file:
              mp2 = pickle.load(file)      

        X_test_new =[k_Symptom_1,k_Symptom_2,k_Symptom_3,k_Symptom_4,k_Symptom_5,k_Symptom_6,k_Symptom_7,k_Symptom_8]
        
        # prediction 1
        datanew=pd.DataFrame(X_test_new).transpose()
        scaler = StandardScaler().fit(X_trainval)
        X_trainval_scaled = scaler.transform(datanew)
        pre=mp.predict(X_trainval_scaled)
        print("Prediction 1",pre[0])

        X_test_new.append(pre[0])

        # prediction 2
        X = data[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF','CDR']]
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size= 0.20, random_state=42, stratify=y)

        datanew=pd.DataFrame(X_test_new).transpose()
        scaler = StandardScaler().fit(X_trainval)
        X_trainval_scaled = scaler.transform(datanew)
        pre2=mp2.predict(X_trainval_scaled)
        print("prediction 2",pre[0])

        res=pd.Series(pre[0]).to_json(orient='values')
        res2=pd.Series(pre2[0]).to_json(orient='values')

        return {"status":True,"msg":res,"prd":res2}
    
@app.route('/predictPneumonia', methods=['GET', 'POST'])
def predict_pneumonia():
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
    app.run(debug=True, port=5000)
