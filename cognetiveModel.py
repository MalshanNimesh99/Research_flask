from flask import Flask,jsonify,request
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from sklearn.preprocessing import scale # scale and center the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
 
app = Flask(__name__)
CORS(app)


# ################################### CKD ##############################################################

@app.route('/ckd',methods=['POST'])
def getdata_ckd():
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
 

if __name__ == '__main__':
 
    
    app.run() 