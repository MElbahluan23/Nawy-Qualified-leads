from calendar import c
from subprocess import HIGH_PRIORITY_CLASS
from flask import Flask, request, jsonify, render_template
import pickle
## for data
import json
import pandas as pd
import numpy as np
import os
from numpy import sort

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set(rc={'figure.figsize': [10, 7]}, font_scale=1.0)

## for processing and feature engineering
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.sparse import csr_matrix,coo_matrix
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import NearMiss
import re
def list_toString(action):
    
    "Function to convert list into strings"
    action = [ str(i) for i in action ]
    
    action = [ re.sub('nan','',i) for i in action ] 
    
    action = ' '.join(action)
    
    return action


def clean_text(row):
    ## clean (convert to lowercase and remove punctuations and characters and then strip,convert from string to list)
    lst_text = simple_preprocess(row)
   
    ## back to string from list
    text = " ".join(lst_text)
    return text
## for processing
from sklearn.utils.class_weight import compute_sample_weight
import nltk.corpus
from gensim.utils import simple_preprocess



app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model
encoder = pickle.load(open('encoder', 'rb')) # loading the encoder
Vectorizer = pickle.load(open('vectorizer', 'rb')) # loading the vectorizer
FeatureImp = pickle.load(open('FeatureImp', 'rb')) # loading the FeatureImp

@app.route('/', methods=['GET']) # Homepage
def home():
    return render_template('index.html')


def list_toString(action):
    
    "Function to convert list into strings"
    action = [ str(i) for i in action ]
    
    action = [ re.sub('nan','',i) for i in action ] 
    
    action = ' '.join(action)
    
    return action

def title_preprocess(title,lst_stopwords):
    ## clean (convert to lowercase and remove punctuations and characters and then strip,convert from string to list)
    lst_text = simple_preprocess(title)
                
    ## back to string from list
    text = " ".join(lst_text)
    return text
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    

    # retrieving values from form
    lead_mobile_network =  str(request.form['lead_mobile_network'])
    method_of_contact =  str(request.form['method_of_contact'])
    lead_source =  str(request.form['lead_source'])
    lead_time_year =  int(request.form['lead_time_year'])
    lead_time_month =  int(request.form['lead_time_month'])
    lead_time_dow =  str(request.form['lead_time_dow'])
    lead_time_hour =  int(request.form['lead_time_hour'])
    campaign =  str(request.form['campaign'])
    ad_group =  str(request.form['ad_group'])
    location =  str(request.form['location'])
    TotalVisits =  int(request.form['TotalVisits'])
    campaing_ad_group_loc = campaign + " "+ad_group +" "+location
    Tfidf=Vectorizer.transform([campaing_ad_group_loc])
    Tfidf=pd.DataFrame(Tfidf.toarray(),columns=np.array(Vectorizer.get_feature_names()))

    cols=['lead_mobile_network', 'method_of_contact', 'lead_source','lead_time_year', 'lead_time_month', 'lead_time_dow', 'lead_time_hour','TotalVisits']
    df=pd.DataFrame(columns=cols)
    df.loc[0]=[lead_mobile_network,method_of_contact,lead_source,lead_time_year,lead_time_month,lead_time_dow,lead_time_hour,TotalVisits]
    
    dropList = ['TotalVisits']
   
    df_ohe=encoder.transform(df.drop(columns=dropList)).toarray()
    df_ohe=pd.DataFrame(df_ohe,columns=np.array(encoder.get_feature_names()))

    DF =pd.concat([Tfidf,df_ohe,df[['TotalVisits']]],axis=1)
    DF=FeatureImp.transform(DF)
    prediction = model.predict(DF) # making prediction

    if prediction ==0:
        pred='High'
    else:
        pred ='Low'


    return render_template('index.html', prediction_text='lead is {} qualified '.format(pred)) # rendering the predicted result

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)