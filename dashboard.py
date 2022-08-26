
from unittest.util import unorderable_list_difference
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

    API_URL = "https://obscure-waters-95734.herokuapp.com/"
    #API_URL = "http://127.0.0.1:5000/"

    background = Image.open("image.png")
    col1, col2, col3 = st.columns([0.2, 5, 0.2])
    col2.image(background, use_column_width=True)

    st.latex(r"""\textbf{Client}""")

    client_id=st.number_input("Please select the ID you want to analyse : ",100001,200000)

    result=""
    result_imp=""
    result_box=""
    result_bar=""
    result_inf=""
    result_desc=""
    result_box_inc=""
    result_box_age=""

    st.latex(r"""\textbf{Client informations}""")

    st.write("This section gives you the basic informations of the client selected and gives you an idea of the client profile. In order to see informations, please select the informations fields bellow")
    check = st.checkbox("Informations")
    if check: 
        result_inf = requests.get(API_URL+"informations?client_id="+str(client_id)).json()
        if result_inf[3]==0:
            result_inf[3] = "F"
        else:
            result_inf[3] = " M"
        
        if result_inf[-1]==0:
            result_inf[-1] = " not married"
        else:
            result_inf[-1] = " married"
        
        #st.write("Income amount : ",result_inf[0])
        #st.write("Annuity : ",result_inf[1])
        #st.write("Credit amount : ",result_inf[2])
        #st.write("Gender : ",result_inf[3])
        #st.write("Age : ",result_inf[4])
        #st.write("Number of children ",result_inf[5])
        #st.write("Married or not ? ",result_inf[6])

         
        df_info = pd.DataFrame({"Income amount : " : [result_inf[0]], "Annuity : " : [result_inf[1]], "Credit amount : " : [result_inf[2]], "Gender : " :result_inf[3] , "Age : " : [result_inf[4]], "Number of children " : [result_inf[5]],"Married or not ? " : [result_inf[6]]})
        st.dataframe(df_info)

        st.latex(r"""{{\color{red}\text{\underline{Client information is marked in red}}}}""")

        result_box_inc = requests.get(API_URL+"boxplot?feature=AMT_INCOME_TOTAL").json()
    
        fig, ax1 = plt.subplots()
        ax1.set_title('Clients income')
        ax1.hist(result_box_inc, bins=20, density= True)
        ax1.axvline(x = result_inf[0], color = 'r', label = 'client income')
        st.pyplot(fig)

        result_box_age = requests.get(API_URL+"boxplot?feature=DAYS_BIRTH").json()
        age = [-(i*100)/(365.25 * 100) for i in result_box_age]

        fig2, ax2 = plt.subplots()
        ax2.set_title('Clients age')
        ax2.hist(age, bins=20, density= True)
        ax2.axvline(x = result_inf[4], color = 'r', label = 'Client age')
        st.pyplot(fig2)


        fig3, ax3 = plt.subplots()
        ax3.set_title('')
        ax3.scatter(age,result_box_inc)
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Income")
        ax3.scatter(result_inf[4],result_inf[0], color = 'r')

        st.pyplot(fig3)
        
    st.latex(r"""\textbf{Credit decisions}""")

    st.write("If you click on the Predict button, the credit decision regarding client data will be provide. The result is based on a machine learning model called based on \"lgbm classifier\" ")
        
    if st.button("Predict"):

        #result=decision(client_id)
        result = requests.get(API_URL+"predict?client_id="+str(client_id)).json()
        #st.write({"api" : api})

        st.write(pd.DataFrame(result, index = ["informations"]))

    st.latex(r"""\textbf{Customer feature importance}""")

    st.write("In this section, you can understand which data is most important for a client, it will help you understand the decision taken by the company to accept or not the credit. Also an analysis of client income and his age is provided.") 

    first_n_var=st.number_input("number of feature you want to analyze",0,len(feats))

    if(st.button("Importance")):
        result_imp= requests.get(API_URL+"importances?client_id="+str(client_id)+"&first_n_var="+str(first_n_var)).json()
        x = list(result_imp.keys())
        y = list(result_imp.values())
        #importances(client_id,first_n_var)
        fig, ax1 = plt.subplots()
        ax1.set_title('Scatter plot')
        ax1.barh(x,y, height = 0.2)
        ax1.invert_yaxis()
        ax1.set_xlim([0, 1.5])
        st.pyplot(fig)

    check_global = st.checkbox("Click here to compare with Feature importance global")
    if check_global:  
        background = Image.open("importance.png")
        col1, col2, col3 = st.columns([0.2, 5, 0.2])
        col2.image(background, use_column_width=True)

    check_desc = st.checkbox("Need help on features descriptions ? ")
    if check_desc:

        result_desc = requests.get(API_URL+"descriptions").json()
        df_desc = pd.DataFrame(result_desc).loc[:,["Row", "Description"]]
        options = st.multiselect('Choose a feature you want to analyse', df_desc.loc[:,"Row"])

        if options:
            df_desc.set_index("Row")
            st.write(df_desc[df_desc["Row"]==options[0]])

if __name__ == '__main__':
    main()