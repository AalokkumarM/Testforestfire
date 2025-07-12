import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle 
ridge_model= pickle.load(open('model/ridge.pkl','rb'))
Standard_Scaler= pickle.load(open('model/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')
@app.route('/predictdata',method=['GET','POST'])
def predict_datapoint():
    if request.method =='POST':
        Temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('RH'))
        WH = float(request.form.get('WH'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = Standard_Scaler.transform([[Temperature, RH, WH, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")