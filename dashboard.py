
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
import cloudpickle
import numba


#app = flask.Flask(__name__)
#app.config["DEBUG"] = False

#Chargement du tableau et du modèle
df = pd.read_pickle("df.gz")
df.drop(columns=["index"], inplace=True)
df.set_index("SK_ID_CURR", inplace=True)
feats = np.genfromtxt('feats.csv', dtype='unicode', delimiter=',')

model = joblib.load("pipeline_credit.joblib")

explainer = joblib.load("pipeline_importance.joblib")


def make_prediction(client_id):
    return model.predict_proba([df[feats].loc[client_id]])[0, 1]
    
def explain(client_id):
    return explainer.shap_values(df[feats].loc[client_id].to_numpy().reshape(1, -1))[1][0][:]


#@app.route('/', methods=['GET'])
def index():
    return {'message': 'Hello, stranger'}


#@app.route('/predict', methods=["GET"])
def decision(client_id):
    #if 'client_id' in request.args:
        #client_id = int(request.args["client_id"])

        pred = make_prediction(client_id)
        if pred>0.27:
            return " The probability is : "+str(pred)+", which is higher than 0.27 (the threshold), the credit is accpeted"

                   #{"The client od is : ": client_id,
                   #"The probability is higher than 0.27, the value is": pred,
                   #"The credit is": "accepted" }
        else:
            return " The probability is : "+str(pred)+", which is bellow than 0.27 (the threshold), the credit is not accpeted"

                   #{"The client od is: ": client_id,
                   #"The probability is bellow than 0.27, the value is ": pred,
                   #"The credit is": "not accepted" }
    #else:

    #   return "Error"  


#@app.route('/importances', methods=["GET"])
def importances(client_id, first_n_var):
    
    #if 'client_id' in request.args:
        res={}
        #client_id = int(request.args["client_id"])
        importance = explain(client_id).tolist()
        
        for key in list(feats):
            for value in importance:
                res[key] = value
                importance.remove(value)
                break  
        res_sorted = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])[::-1]}
        return json.dumps(dict(itertools.islice(res_sorted.items(), first_n_var)))
    #else:
    #    return "Error"   

    

def scatter(feature_1, feature_2):
    data_x = np.array(np.sort(df[feature_1]))
    data_y = np.array(np.sort(df[feature_2]))
    fig2, ax2 = plt.subplots()
    ax2.set_title('Scatter plot')
    ax2.scatter(data_x,data_y)
    st.pyplot(fig2)



def main():
    
    st.markdown("<h1 style='text-align: center; color: grey;'>Projet 7 - Open Classroom </h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'> Credit scoring </h2>", unsafe_allow_html=True)

    #API_URL = "http://127.0.0.1:5000/api/"


    background = Image.open("image.png")
    col1, col2, col3 = st.columns([0.2, 5, 0.2])
    col2.image(background, use_column_width=True)

    client_id=st.number_input("client_id",100000,200000)
    first_n_var=st.number_input("first_n_var",0,len(feats))

    result=""
    result_imp=""
    result_box=""
    result_bar=""

    if st.button("Predict"):
        #result=decision(client_id)
        result = requests.get("http://127.0.0.1:5000/predict?client_id="+str(client_id)).json()
        #st.write({"api" : api})
    st.success(result)

    if(st.button("Importance")):
        result_imp= requests.get("http://127.0.0.1:5000/importances?client_id="+str(client_id)+"&first_n_var="+str(first_n_var)).json()
        x = list(result_imp.keys())
        y = list(result_imp.values())
        #importances(client_id,first_n_var)
        fig, ax1 = plt.subplots()
        ax1.set_title('Scatter plot')
        ax1.barh(x,y, height = 0.2)
        ax1.invert_yaxis()
        ax1.set_xlim([0, 1.5])
        st.pyplot(fig)


    
    if (st.button("Distribution")):
        result_box = requests.get("http://127.0.0.1:5000/boxplot?feature=AMT_CREDIT").json()
        fig, ax1 = plt.subplots()
        ax1.set_title('AMT_CREDIT')
        ax1.hist(result_box, bins=20, density= True)
        st.pyplot(fig)
     
    #st.write(pd.DaraFrame(result_imp))

    #options = st.multiselect('Choose a feature you want to analyse', list(feats))
    #st.write('You selected:', options)

    page_names = ['Univariate', "Bivariate"]
    page = st.radio("Navigation",page_names)
    st.write(" You choose :",page)

    if page == 'Univariate':

        features = st.multiselect('Choose a feature you want to analyse', list(feats))
        result_box = requests.get("http://127.0.0.1:5000/boxplot?feature="+str(features[0])).json()
        fig, ax1 = plt.subplots()
        ax1.set_title('Boxplot plot')
        ax1.boxplot(result_box)
        st.pyplot(fig)

    else: 
        st.write("OK")
        
     

if __name__ == '__main__':
    main()