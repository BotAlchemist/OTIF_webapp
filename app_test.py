# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 08:45:26 2021

@author: Sumit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import  roc_curve, roc_auc_score
import os

from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings,ContainerClient
import pickle
import json
st.set_option('deprecation.showPyplotGlobalUse', False)


html_header='''
<div style= 'background-color: pink; padding:13px';>
<h2 style= "color:black; text-align:center;"><b> On Time In Full: Prediction Results </b></h2>
</div>
<br>
'''

st.markdown(html_header, unsafe_allow_html=True)


files= [f for f in os.listdir('.') if os.path.isfile(f)]
files= list(filter(lambda f: f.endswith('.csv'), files))

filename_select = st.sidebar.selectbox('Select dataset *', files)

df= pd.read_csv(filename_select)

if st.sidebar.checkbox("Show dataset", False):
    st.write(df.head(5))
    
if st.sidebar.checkbox("Run prediction", False): 
    
    ################################################ GET META DATA ################################
    blob = BlobClient(account_url="https://sumitfilestorage.blob.core.windows.net",
                      container_name="model-container",
                      blob_name="OTIF_meta_data.txt",
                      credential="4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==")
    
    with open("OTIF_meta_data.txt", "wb") as f:
        data = blob.download_blob()
        data.readinto(f)
    
    with open('OTIF_meta_data.txt') as json_file:
        data = json.load(json_file)
        column_names= data['Columns']
        algo_name= data['Algorithm']
        metrics= data['Metrics']
        parameters= data['Parameters']

    created_on_column = parameters['Created on']
    req_date_column = parameters['Req delivery date']
    order_qty = parameters['Order received']
    key_columns = parameters['Keys'] 
    columns= data['Columns']
    
    ################################################ GET ENCODER ################################
    blob = BlobClient(account_url="https://sumitfilestorage.blob.core.windows.net",
                      container_name="model-container",
                      blob_name="OTIF_encode.txt",
                      credential="4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==")
    
    with open("OTIF_encode.txt", "wb") as f:
        encoder = blob.download_blob()
        encoder.readinto(f)
    with open('OTIF_encode.txt') as json_file:
        encoder = json.load(json_file)
        
    ################################################ GET MODEL ################################
    blob = BlobClient(account_url="https://sumitfilestorage.blob.core.windows.net",
                      container_name="model-container",
                      blob_name="OTIF_model.sav",
                      credential="4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==")
    with open("OTIF_model.sav", "wb") as f:
        model = blob.download_blob()
        model.readinto(f)
        
    loaded_model = pickle.load(open("OTIF_model.sav", 'rb')) 
    

    df[created_on_column]= pd.to_datetime(df[created_on_column], errors= "coerce")
    df[req_date_column]= pd.to_datetime(df[req_date_column], errors= "coerce")
    df['Day_name'] = df[created_on_column].dt.day
    df['Week_name'] = df[created_on_column].dt.dayofweek
    df['Month_name'] = df[created_on_column].dt.month
    df['Quarter_name'] = df[created_on_column].dt.quarter
    df['Weekend'] = np.where((df[created_on_column].dt.dayofweek)% 5, 0,1)
    df['Planned interval'] = (df[req_date_column] - df[created_on_column]).dt.days

    st.write(df)
    
    final_columns= []
    for col in df.columns:
        if col in columns:
            final_columns.append(col)
            
    
    df_test= df.copy()
    df_test= df_test[final_columns]
    
    object_columns=[]
    for col in df_test.columns:
        if df_test[col].dtype=='object':
            object_columns.append(col)
    
    
    for col in object_columns:
        df_test= df_test.replace({col:encoder[col]})
        
    y_pred= (loaded_model.predict_proba(df_test)[::,1] >= 0.5).astype('int')
    y_prob= loaded_model.predict_proba(df_test)[::,1]
    
    prob_val= []
    for i in range(len(y_prob)):
        if y_pred[i] == 1:
            val = y_prob[i]
        else:
            val = 1- y_prob[i]
    
        prob_val.append(val)
    
    prob_val= [round(i*100,2) for i in prob_val]
    
    
    
    
    
    
    
    
    
    
    
    st.write(df_test)