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
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import base64

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

df_original= pd.read_csv(filename_select)

df= df_original.copy()

if st.sidebar.checkbox("Show dataset", False):
    st.write(df.head(5))

if st.sidebar.checkbox("Show last training parameters", False):
    st.markdown("### Algorithm: " + str(algo_name))
    st.markdown("#### Accuracy: " + str(metrics['Accuracy']) + " %")
    st.markdown("#### ")


if st.sidebar.checkbox("Run prediction", False): 
    with st.spinner('Running prediction...'):

        df[created_on_column]= pd.to_datetime(df[created_on_column], errors= "coerce")
        df[req_date_column]= pd.to_datetime(df[req_date_column], errors= "coerce")
        df['Day_name'] = df[created_on_column].dt.day
        df['Week_name'] = df[created_on_column].dt.dayofweek
        df['Month_name'] = df[created_on_column].dt.month
        df['Quarter_name'] = df[created_on_column].dt.quarter
        df['Weekend'] = np.where((df[created_on_column].dt.dayofweek)% 5, 0,1)
        df['Planned interval'] = (df[req_date_column] - df[created_on_column]).dt.days
        
        
        
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
        df_original['OTIF Prediction'] = y_pred
        df_original['Prediction confidence'] = prob_val
        df_original['OTIF Prediction Label'] = df_original['OTIF Prediction'].replace([0,1], ['Not OTIF', 'OTIF'])
        
        ################################################## SHOWING DASHBOARD #######################################

        
        
        
        df_original[created_on_column]= pd.to_datetime(df_original[created_on_column], errors= "coerce")
        df_original= df_original.sort_values([created_on_column])
        df_original= df_original.set_index(created_on_column)
        
       
        pred_confidence= st.slider("Prediction confidence threshold greater than:", 50, 99)
        df_original= df_original[df_original['Prediction confidence']> pred_confidence]
        
        
        drop_columns= st.multiselect('Select columns to be dropped from display', df_original.columns.tolist())
        df_original= df_original.drop(drop_columns, axis=1)
        
        st.write(df_original)
        
        no_otif_records= len(df_original[df_original['OTIF Prediction']==1])
        no_not_otif_records= len(df_original[df_original['OTIF Prediction']==0])
        st.success("Number of orders predicted to be On Time & In Full: "+ str(no_otif_records))
        st.warning("Number of orders predicted to be NOT On Time & In Full: "+ str(no_not_otif_records))
        
        
        ############################################ DISPLAY PIE CHART ################################
        pie_labels= ['OTIF', 'Not OTIF']
        pie_values= [no_otif_records, no_not_otif_records]

        fig = px.pie(values=pie_values, names= pie_labels)
        st.plotly_chart(fig)
        
        
        download=st.button('Download Result')
        if download:
            csv = df_original.to_csv()
            b64 = base64.b64encode(csv.encode()).decode()
            linko= f'<a href="data:file/csv;base64,{b64}" download="OTIF prediction.csv">Click to download the file</a>'
            st.markdown(linko, unsafe_allow_html=True)
        
        
    
    
    
    
    
    
    
    
    
    
    #st.write(df_test)