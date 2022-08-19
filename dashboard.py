
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