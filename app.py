# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:14:40 2021

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
<h2 style= "color:black; text-align:center;"><b> On Time In Full: Training app </b></h2>
</div>
<br>
'''

st.markdown(html_header, unsafe_allow_html=True)
#st.title('On Time In Full Training Web App')



files= [f for f in os.listdir('.') if os.path.isfile(f)]
files= list(filter(lambda f: f.endswith('.csv'), files))

filename_select = st.sidebar.selectbox('Select dataset *', files)

df= pd.read_csv(filename_select)

if st.sidebar.checkbox("Show dataset", False):
    st.write(df.head(5))

created_on_column= st.sidebar.selectbox('Select created on column *', df.columns.tolist())
req_date_column= st.sidebar.selectbox('Select requested date column *', df.columns.tolist())
delivery_date_column= st.sidebar.selectbox('Select delievry date column *', df.columns.tolist())

date_extractor_options= ['Day', 'Week','Weekend', 'Month', 'Quarter']
date_extractor= st.sidebar.multiselect('Select date components to be extracted', date_extractor_options) 
#target_column= st.sidebar.selectbox('Select target column', df.columns.tolist())



order_qty= st.sidebar.selectbox('Select order quantity column *', df.columns.tolist())
delivered_qty= st.sidebar.selectbox('Select delivered quantity column *', df.columns.tolist())

drop_columns= st.sidebar.multiselect('Select columns to be dropped ', df.columns.tolist())

key_columns= st.sidebar.multiselect('Select key columns *', df.columns.tolist()) 


column_parameters={}
column_parameters['Created on'] = created_on_column
column_parameters['Req delivery date'] = req_date_column
column_parameters['Delivered Date'] = delivery_date_column
column_parameters['Order received'] = order_qty
column_parameters['Order deivered'] = delivered_qty
column_parameters['Keys'] = key_columns



################################################################## Pre-processing Functions##############################

def get_on_time(row):
    if row[req_date_column] == row[delivery_date_column]:
        val=0
    elif row[req_date_column] < row[delivery_date_column]:
        val= 1
    else:
        val= -1
    return val


def get_otif(row):
    if row['In Full'] == 1 and row['On Time']==0:
        val=1
    else:
        val=0
    return val

def preprocess_df(df,order_qty, delivered_qty,req_date_column,delivery_date_column,key_columns, created_on_column  ):
    df= df.drop(key_columns, axis=1)
    #with st.spinner('Calculating In Full...'):
    df['In Full'] = np.where(df[order_qty] == df[delivered_qty], 1,0)
    
    #with st.spinner('Calculating On Time...'):
    df['On Time'] = df.apply(get_on_time, axis=1)
   
    #with st.spinner('Calculating OTIF...'):
    df['OTIF'] = df.apply(get_otif, axis=1)
    
    df['Planned interval'] = (df[req_date_column] - df[created_on_column]).dt.days
    
    
    object_columns=[]
    for col in df.columns:
        if df[col].dtype=='object':
            object_columns.append(col)
            
    
    encode_dict={}
    for col in object_columns:
        col_unique= df[col].unique()
        col_id= list(range(len(col_unique)))
        encode_dict[col]= dict(zip(col_unique, col_id))
    
    for col in object_columns:
        df= df.replace({col:encode_dict[col]})
    
    return df, encode_dict

############################################# Display results #############################################
def display_results(test_df,prediction,  target, proba):
    html_result='''
    <div style= 'background-color: pink; padding:13px';>
    <h3 style= "color:black; text-align:center;"> Results</h3>
    </div>
    <br>
    '''

    st.markdown(html_result, unsafe_allow_html=True)
    #st.subheader("Results")
    st.success("#### Accuracy: "+ str(round(accuracy_score(test_df[target], prediction)*100,2)) + " %")
    fpr, tpr, _ = roc_curve(test_df[target],  proba)
    auc = roc_auc_score(test_df[target],  proba)
    st.success("#### AUC score: "+ str(round(auc,2)))
    st.text("Classification Report: \n "+  classification_report(test_df[target], prediction))
    
    cm=confusion_matrix(test_df[target], prediction)
    #st.write('Confusion matrix: ', cm)
    
    
    
    display_matrix= st.radio("Matrix:", ("Confusion Matrix", "AUC score"))
    if display_matrix == "Confusion Matrix":
    
        fig, ax = plt.subplots(figsize=(2,2))
        sn.heatmap(cm, annot=True, ax=ax)
        st.write(fig)
    else:   

        fig, ax = plt.subplots(figsize=(2,2))
        plt.plot(fpr,tpr,label="AUC curve, auc="+str(auc))
        st.write(fig)
        
    return round(accuracy_score(test_df[target], prediction)*100,2), auc
   


############################################################ Submit model header ##########################################
def show_submit_header():
     html_submit='''
     <div style= 'background-color: pink; padding:13px';>
     <h3 style= "color:black; text-align:center;"> Click on below button to submit the model </h3>
     </div>
     <br>
     '''

     st.markdown(html_submit, unsafe_allow_html=True)
     
############################################################ Save Resuts ###################################################
def save_model_to_blob(model, train_columns, algo_name, encode_dict, accuracy, auc, column_parameters):
    local_path = "." 
    filename = 'OTIF_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    upload_file_path = os.path.join(local_path, filename)
    my_connection_string="DefaultEndpointsProtocol=https;AccountName=sumitfilestorage;AccountKey=4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==;EndpointSuffix=core.windows.net"
    model_container= "model-container"
    blob_service_client= BlobServiceClient.from_connection_string(my_connection_string)
    blob_client= blob_service_client.get_blob_client(container= model_container, blob= filename)
    
    
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        st.success("Model saved to Azure Blob Storage" )
        
        
    meta_data={}
    meta_data['Columns']= train_columns
    meta_data['Metrics']= {
    'Accuracy': round(accuracy,2),
    'AUC score': round(auc,2)}
    meta_data['Algorithm']= algo_name
    meta_data['Parameters']= column_parameters
    
    
    
    with open('OTIF_meta_data.txt', 'w') as outfile:
        json.dump(meta_data, outfile)
    
        
    filename = 'OTIF_meta_data.txt'
    upload_file_path = os.path.join(local_path, filename)
    blob_client= blob_service_client.get_blob_client(container= model_container, blob= filename)
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        st.success("Meta Data saved to Azure Blob Storage" )
        
    
    with open('OTIF_encode.txt', 'w') as outfile:
        json.dump(encode_dict, outfile)
    
        
    filename = 'OTIF_encode.txt'
    upload_file_path = os.path.join(local_path, filename)
    blob_client= blob_service_client.get_blob_client(container= model_container, blob= filename)
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        st.success("Encoders saved to Azure Blob Storage" )
    
    
    
    
    
    
    
    
    

if len(filename_select) !=0 and len(key_columns)>0 and created_on_column != req_date_column and created_on_column != delivery_date_column and order_qty != delivered_qty:

    df=df.dropna()
    
    if len(drop_columns)!=0:
        df= df.drop(drop_columns, axis=1)
    
    date_columns= [created_on_column, req_date_column,delivery_date_column]
    for col in date_columns:
        df[col]= pd.to_datetime(df[col])
    
    df= df[df[created_on_column] < df[delivery_date_column]]
    if 'Day' in date_extractor:
        df['Day_name'] = df[created_on_column].dt.day
    if 'Week' in date_extractor:  
        df['Week_name'] = df[created_on_column].dt.dayofweek
    if 'Month' in date_extractor:
        df['Month_name'] = df[created_on_column].dt.month
    if 'Quarter' in date_extractor:
        df['Quarter_name'] = df[created_on_column].dt.quarter
    if 'Weekend'  in date_extractor:
        df['Weekend'] = np.where((df[created_on_column].dt.dayofweek)% 5, 0,1)
    
        
    df= df.sort_values([created_on_column])
    
     
    try:
        df= df[df[order_qty] >= df[delivered_qty]] 
    except:
        st.warning("Please select parameters correctly")         
       

    df, encode_dict= preprocess_df(df,order_qty, delivered_qty,req_date_column,delivery_date_column,key_columns ,created_on_column )    
    df = df.set_index(created_on_column)
       
    next_run= st.radio("Run", ( "Model", "Analysis")) 

    if  next_run== "Analysis":
        st.info("Correlation matrix")
        corr = df.corr()
        fig= corr.style.background_gradient(cmap='coolwarm')
        st.write(fig)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    if next_run=="Model":
        split= st.slider("Train/Test split %", 50, 99)
        
        df= df.drop([delivered_qty,req_date_column,delivery_date_column], axis=1)
        df= df.drop(['In Full', 'On Time'], axis=1)
        
        split_value= int((split/100)*len(df))
        train_df= df.iloc[:split_value]
        test_df= df.iloc[split_value:]
        
        st.markdown("##### Length of Training set: "+ str(len(train_df)))
        st.markdown("##### Length of Test set: "+ str(len(test_df)))
        
        html_choose_algo='''<br>
        <div style= 'background-color: pink; padding:13px';>
        <h3 style= "color:black; text-align:center;"> Choose algorithm </h3>
        </div>
        <br>
        '''
    
        st.markdown(html_choose_algo, unsafe_allow_html=True)
        
        algorithm_name_list= ['None',   'Random Forest','XG Boost', 'K Nearest Neighbor']
        algorithm_name= st.selectbox(' ', algorithm_name_list)
        

        
        if algorithm_name == "None":
            st.warning("Please select an algorithm")
        
        ####################################### Random Forest #######################################
        if algorithm_name == "Random Forest":
 
            run_default= st.radio("Use default settings?",('Yes', 'No') )
            
            
            
            
            if run_default == 'Yes':
                model= RandomForestClassifier()
            else:
                choose_n_estimators= st.slider("No. of trees", 10, 500)
                choose_max_depth= st.slider("Max. depth", 2, 20)
                choose_max_features= st.selectbox("Max features", ['auto', 'sqrt', 0.2])
                choose_min_sample_leaf = st.slider("Min Sample Leaf",1, 100)
                
                
                model= RandomForestClassifier(n_estimators=choose_n_estimators, max_depth= choose_max_depth,
                                                max_features=choose_max_features,
                                                min_samples_leaf=choose_min_sample_leaf)
            
            with st.spinner('Training the model...'):
                model.fit(train_df.drop("OTIF", axis=1), train_df["OTIF"])
            
            rf_prediction = model.predict(test_df.drop("OTIF", axis=1))
            rf_prediction_proba = model.predict_proba(test_df.drop("OTIF", axis=1))[::,1]
            
            accuracy, auc= display_results(test_df,rf_prediction,  "OTIF", rf_prediction_proba)
            
            show_submit_header()
            if (st.button("Submit Model")):
                save_model_to_blob(model, train_df.columns.tolist(), 'Random Forest',encode_dict,accuracy, auc, column_parameters)
            
            
        ####################################### KNN #######################################
        elif algorithm_name == 'K Nearest Neighbor':
            best_k= st.slider("Select the value of K", 1, 20)
            model= KNeighborsClassifier(n_neighbors= best_k)
            with st.spinner('Training the model...'):
                model.fit(train_df.drop("OTIF", axis=1), train_df["OTIF"])
            
            knn_prediction = model.predict(test_df.drop("OTIF", axis=1))
            knn_prediction_proba = model.predict_proba(test_df.drop("OTIF", axis=1))[::,1]
            
            accuracy, auc= display_results(test_df,knn_prediction,  "OTIF", knn_prediction_proba)
            
            show_submit_header()
            if (st.button("Submit Model")):
                save_model_to_blob(model, train_df.columns.tolist(), 'K Nearest Neighbor',encode_dict,accuracy, auc, column_parameters)
            
    ################################################### XG Boost ##############################
        elif algorithm_name == 'XG Boost':
            run_default= st.radio("Use default settings?",('Yes', 'No') )
            if run_default == 'Yes':
                model= XGBClassifier()
            else:
                choose_n_estimators= st.slider("No. of trees", 10, 500)
                choose_booster= st.selectbox("Booster", ['gbtree', 'gblinear'])
                choose_max_depth= st.slider("Max. depth", 2, 20)
                choose_learning_rate= st.selectbox("Learning Rate", [0.01, 0.1, 0.2,0.5])
                choose_min_sample_leaf = st.slider("Min Sample Leaf",1, 100)
                
                model= XGBClassifier(n_estimators=choose_n_estimators, max_depth= choose_max_depth,
                                                booster= choose_booster, learning_rate= choose_learning_rate,
                                                min_samples_leaf=choose_min_sample_leaf)
            
            with st.spinner('Training the model...'):
                model.fit(train_df.drop("OTIF", axis=1), train_df["OTIF"])
                
            xgb_prediction = model.predict(test_df.drop("OTIF", axis=1))
            xgb_prediction_proba = model.predict_proba(test_df.drop("OTIF", axis=1))[::,1]
            accuracy, auc= display_results(test_df,xgb_prediction,  "OTIF",xgb_prediction_proba)   
            
            show_submit_header()
            if (st.button("Submit Model")):
                save_model_to_blob(model, train_df.columns.tolist(), 'XG Boost',encode_dict,accuracy, auc, column_parameters)
                

else:
    st.info("Please select the parameters to train the model")    
    
