#pip install shap

#import flask
import joblib
import pandas as pd
import shap
import json
import numpy as np
import lightgbm
import streamlit as st 

#app = flask.Flask(__name__)
#app.config["DEBUG"] = False

#Chargement du tableau et du modèle
df = pd.read_pickle("df.gz")
df.drop(columns=["index"], inplace=True)
df.set_index("SK_ID_CURR", inplace=True)
feats = np.genfromtxt('feats.csv', dtype='unicode', delimiter=',')

model = joblib.load("pipeline_credit.joblib")

explainer = joblib.load("shap_explainer.joblib")

def make_prediction(client_id):
    return model.predict_proba([df[feats].loc[client_id]])[0, 1]
    
def explain(client_id):
    return explainer.shap_values(df[feats].loc[client_id].to_numpy().reshape(1, -1))[1][0][:]


#@app.route('/', methods=['GET'])
def index():
    return {'message': 'Hello, stranger'}


if __name__=='__main__':
    main()

app.run()