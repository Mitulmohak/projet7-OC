#pip install shap

import flask
from flask import request, jsonify
import joblib
import pandas as pd
import shap
import json
import numpy as np
import lightgbm
import streamlit as st 
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import requests



app = flask.Flask(__name__)
app.config["DEBUG"] = False

#Chargement du tableau et du modèle
df = pd.read_pickle("df.gz")
df_desc = pd.read_pickle("df_desc.gz")
df.drop(columns=["index"], inplace=True)
df.set_index("SK_ID_CURR", inplace=True)
feats = np.genfromtxt('feats.csv', dtype='unicode', delimiter=',')

model = joblib.load("pipeline_credit.joblib")

explainer = joblib.load("pipeline_importance.joblib")


def make_prediction(client_id):
    return model.predict_proba([df[feats].loc[client_id]])[0, 1]
    
def explain(client_id):
    return explainer.shap_values(df[feats].loc[client_id].to_numpy().reshape(1, -1))[1][0][:]


@app.route('/', methods=['GET'])
def index():
    return {'message': ' This is a credit scoring app ! '}


@app.route('/predict', methods=["GET"])
def proba():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = make_prediction(client_id)
        if pred<0.84:
        	return {"The client id is" : [client_id],
          		   "The probability is bellow than 0.84, the value is": [pred],
          		   "The credit is": ["accepted"] }
        else:
        	return {"The client id is: ": [client_id],
          		   "The probability is higher than 0.84, the value is ": [pred],
          		   "The credit is": ["not accepted"] }
    else:

    	return "Error"  

@app.route('/informations', methods = ["GET"])
def info():

    client_id = int(request.args["client_id"])
    amt_inc = df['AMT_INCOME_TOTAL'].loc[client_id]
    amt_ann = df['AMT_ANNUITY'].loc[client_id]
    amt = df['AMT_CREDIT'].loc[client_id]
    gender = df['CODE_GENDER'].loc[client_id]
    age = int(df['DAYS_BIRTH'].loc[client_id] *100)/(365.25 * 100)
    cnt_child = df['CNT_CHILDREN'].loc[client_id]
    status = df['NAME_FAMILY_STATUS_Married'].loc[client_id]
    return json.dumps([int(amt_inc),int(amt_ann),int(amt),int(gender),int(-age),int(cnt_child),int(status)])

@app.route('/importances', methods=["GET"])
def importances():
    
    if ('client_id' and 'first_n_var') in request.args:
       res={}
       client_id = int(request.args["client_id"])
       first_n_var = int(request.args["first_n_var"])
       importance = explain(client_id).tolist()
        
       for key in list(feats):
           for value in importance:
               res[key] = value
               importance.remove(value)
               break  
       res_sorted = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])[::-1]}
       return json.dumps(dict(itertools.islice(res_sorted.items(), first_n_var)))
    else:
       return "Error"   


@app.route("/descriptions", methods = ["GET"])
def description():    
    return json.dumps(df_desc.to_dict())


@app.route('/boxplot', methods=["GET"])
def boxplot():
    if 'feature' in request.args:
       
        feature = request.args["feature"]
        data = df[feature]
        return json.dumps(data.tolist())
  
    else:

        return "Error" 


@app.route('/barplot', methods=["GET"])
def barplot():
    if 'feature' in request.args:
        feature = request.args["feature"]
        data = df[feature]
        return json.dumps(data.tolist())
  
    else:
        return "Error"


if __name__ == '__main__':
    app.debug = True
    app.run()