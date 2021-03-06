#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import re
from IPython.display import display

import numpy as np
import shutil
import posixpath
import plotly.graph_objects as go

# EDA Pkgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
import glob
import seaborn as sns
import datetime
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import sys
import os

#import tensorflow as tf
import keras


# In[3]:

st.title('Welcome to StressWatch')

st.header('Predicting Stress in Early Stages')


st.subheader('Please select from the sidebar options')   



@st.cache
def ReadSignals():
    acc_eda_temp_dic = {}
    files = [file for file in glob.glob('stress_data/Acc_Temp_EDA/*.csv')]
    column_labels = ["Hr", "Min", "Sec", "Accz", "Accy", "Accx", "Temp", "EDA", "Label"]
    #subjects = set(file.split('/')[-1].split('Acc'))
    for subject in files:
        file = pd.read_csv(subject)#, names = column_labels)
        file.drop(labels = ['Hour', 'Minute', 'Second'], axis =1, inplace = True)
        
        # prepare target
        le = LabelEncoder()
        le.fit(file['Label'])
        file ['Label'] = le.transform(file['Label'])
        acc_eda_temp_dic[subject.split('/')[2].split('Acc')[0]] = file
        
    hr_o2_dic = {}
    files2 = [file for file in glob.glob('stress_data/Hr_O2/*.csv')]
    column_labels2 = ["Hr", "Min", "Sec", "Heart rate", "SpO2", "Label"]
    #subjects = set(file.split('/')[-1].split('Acc'))
    for subject in files2:
        file = pd.read_csv(subject)#, names = column_labels)
        file.drop(labels = ['Hour', 'Minute', 'Second', 'Label'], axis =1, inplace = True)
        file.drop(file.tail(5).index, inplace = True)
        hr_o2_dic[subject.split('/')[2].split('Sp')[0]] = file
        
        
    keys = list(acc_eda_temp_dic.keys())
    rsmpl_dic = {}
    for key in keys:
        rsmpl_dic[key] = acc_eda_temp_dic[key].apply(lambda x: x.iloc[np.r_[0:len(x):8, -1]])
    
    
    rsmpl_reindx_dic = {}
    for key in keys:
        rsmpl_reindx_dic[key] = rsmpl_dic[key].reset_index(drop = True)
    
        
    signal_data = {}
    for key in (rsmpl_reindx_dic.keys() | hr_o2_dic.keys()):
        if key in rsmpl_reindx_dic:
        	signal_data.setdefault(key, []).append(rsmpl_reindx_dic[key])
        if key in hr_o2_dic:
        	signal_data.setdefault(key, []).append(hr_o2_dic[key])
    
    
    return signal_data

@st.cache
def resampled_signals():
	acc_eda_temp_dic = {}
	files = [file for file in glob.glob('stress_data/Acc_Temp_EDA/*.csv')]
	column_labels = ["Hr", "Min", "Sec", "Accz", "Accy", "Accx", "Temp", "EDA", "Label"]
	#subjects = set(file.split('/')[-1].split('Acc'))
	for subject in files:
	    file = pd.read_csv(subject)#, names = column_labels)
	    file.drop(labels = ['Hour', 'Minute', 'Second'], axis =1, inplace = True)

	    # prepare target
	    le = LabelEncoder()
	    le.fit(file['Label'])
	    file ['Label'] = le.transform(file['Label'])

	    numerical_feat = ['AccZ', 'AccY', 'AccX', 'Temp', 'EDA']
	    scaler = StandardScaler()
	    file[numerical_feat] = scaler.fit_transform(file[numerical_feat])

	    acc_eda_temp_dic[subject.split('/')[2].split('Acc')[0]] = file

	hr_o2_dic = {}
	files2 = [file for file in glob.glob('stress_data/Hr_O2/*.csv')]
	column_labels2 = ["Hr", "Min", "Sec", "Heart rate", "SpO2", "Label"]
	#subjects = set(file.split('/')[-1].split('Acc'))
	for subject in files2:
	    file = pd.read_csv(subject)#, names = column_labels)
	    file.drop(labels = ['Hour', 'Minute', 'Second', 'Label'], axis =1, inplace = True)
	    file.drop(file.tail(5).index, inplace = True)

	    feat = ['HeartRate', 'SpO2']
	    file[feat] = scaler.fit_transform(file[feat])

	    hr_o2_dic[subject.split('/')[2].split('Sp')[0]] = file

	keys = list(acc_eda_temp_dic.keys())
	rsmpl_dic = {}
	for key in keys:
	    rsmpl_dic[key] = acc_eda_temp_dic[key].apply(lambda x: x.iloc[np.r_[0:len(x):8, -1]])


	rsmpl_reindx_dic = {}
	for key in keys:
	    rsmpl_reindx_dic[key] = rsmpl_dic[key].reset_index(drop = True)


	signal_data = {}
	for key in (rsmpl_reindx_dic.keys() | hr_o2_dic.keys()):
	    if key in rsmpl_reindx_dic: 
	    	signal_data.setdefault(key, []).append(rsmpl_reindx_dic[key])

	    if key in hr_o2_dic: 
	    	signal_data.setdefault(key, []).append(hr_o2_dic[key])

	return signal_data
	

# In[4]:

# Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading data...')
# Load the participants signals into the dictionary.

signals = ReadSignals()
signals_p = resampled_signals()
# Notify the reader that the data was successfully loaded.
#data_load_state.text("Done! (using st.cache)")


activities = ['View Raw Signals' ,'View Prediction Results']
choices = st.sidebar.selectbox("Select Activity",activities)

if choices == 'View Raw Signals':
	st.subheader("Raw data")
	#signals = ReadSignals()
	user = st.sidebar.selectbox('Please select your subject ID or upload your data', list(signals.keys()), 0)
	
	st.sidebar.file_uploader(label = 'Please upload your signals data in CSV format', type = 'csv')
	
	'You have selected: ', user
	
	st.line_chart(signals[user][0]['EDA'])
	st.line_chart(signals[user][0]['Temp'])
	st.line_chart(signals[user][1]['HeartRate'])
	st.line_chart(signals[user][1]['SpO2'])
	st.line_chart(signals[user][0]['AccZ'])
	st.line_chart(signals[user][0]['AccY'])
	st.line_chart(signals[user][0]['AccX'])

# In[6]:

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

if choices == 'View Prediction Results':
	#st.subheader("Likelihood of being in each state")
	#signals_p = resampled_signals()
	#signals_p = ReadSignals()
	user = st.sidebar.selectbox('Please select your subject ID', list(signals_p.keys()), 0)
	#st.sidebar.file_uploader(label = 'Please upload your signals data in CSV format', type = 'csv')	

	st.write('You have selected: ', user)	
	
	'This plot shows the transition of your affective state over time'
	
	# get data		
		
	@st.cache
	def create_data(user, N_samples):
		length = signals_p[user][0]['EDA'].shape[0]
		max_interval = length//N_samples
		#for i in range(max_interval): 
		#i = np.random.choice(max_interval - 1, 1, replace=True)[0]
		
		user_x = []
		user_y = []
		for i in range(max_interval):
		
			x = [np.hstack(signals_p[user][0]['AccZ'][i*N_samples:(i+1)*N_samples]), 
			    np.hstack(signals_p[user][0]['AccY'][i*N_samples:(i+1)*N_samples]),
			    np.hstack(signals_p[user][0]['AccX'][i*N_samples:(i+1)*N_samples]),
			    np.hstack(signals_p[user][0]['Temp'][i*N_samples:(i+1)*N_samples]),
			    np.hstack(signals_p[user][0]['EDA'][i*N_samples:(i+1)*N_samples]),
			    np.hstack(signals_p[user][1]['HeartRate'][i*N_samples:(i+1)*N_samples]),
			    np.hstack(signals_p[user][1]['SpO2'][i*N_samples:(i+1)*N_samples])]

			y = np.vstack(signals_p[user][0]['Label'][i*N_samples:(i+1)*N_samples])
			
			user_x.append(x)
			user_y.append(y[-1])
						
		return user_x, np.vstack(user_y)
		
	@st.cache
	def batch_generator(batch_size, N_samples):
		#while True:

		batch = [create_data(user, N_samples)]

		batch_X = np.vstack([batch[i][0] for i in range(len(batch))])
		batch_Y = np.vstack([batch[i][1] for i in range(len(batch))])
		
		# yield batch_X, batch_Y
		return batch_X, batch_Y

	load_prediction_models = open("lstm_model.pkl", 'rb') 
	predictor = joblib.load(load_prediction_models)
	load_prediction_models.close()
	
	x, y = batch_generator(batch_size = 16, N_samples=240)
	prediction = predictor.predict(x)*100
	column_names = ['Cognitive Stress', 'Emotional Stress', 'Physical Stress', 'Relax']
	df1 = pd.DataFrame(prediction, columns=column_names)
	#freq = 4
	df1['Time'] = pd.date_range(end='now', periods=len(df1), freq='4min')
	#df1['Time'] = pd.date_range(end='now', periods=8, freq='4min')
	df1.Time = pd.to_datetime(df1.Time, format='%H:%M')
	df1.set_index(['Time'],inplace=True)
	colors = ['orange','red','navy','green']
	
#	fig_line, ax1 = plt.subplots(figsize = (24,15), dpi = 600) 
	#ax1 = sns.lineplot(data = df1), hue = df1.columns)
#	ax1 = sns.lineplot(data = df1['Relax'], ls = '-', color ='red', label = 'Relax')
#	ax1 = sns.lineplot(data = df1['Cognitive Stress'], ls = '-', color ='navy', label = 'Cognitive Stress')
#	ax1 = sns.lineplot(data = df1['Emotional Stress'], ls = '-', color ='orange', label = 'Emotional Stress')
#	ax1 = sns.lineplot(data = df1['Physical Stress'], ls = '-', color ='green', label = 'Physical Stress')
	ax1 = df1.plot(kind='area', stacked=False, alpha=0.3, figsize=(25,15), color = colors)	
	ax1.set_xlim(df1.index[0], df1.index[-1])
	ax1.legend(bbox_to_anchor=(1,1.02), fontsize=34)
	ax1.tick_params(axis='both', which='major', labelsize=34)
	ax1.tick_params(axis='both', which='minor', labelsize=24)
	ax1.tick_params(axis='x', which='major', rotation=90, pad=1)
	ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=4))
	ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
	ax1.set_title('Affective state prediction', fontsize = 40)
	ax1.set_xlabel("Time (H:Min)", fontsize=34, labelpad = 15)
	ax1.set_ylabel("Prediction confidence %", fontsize=34)
	plt.rcParams["figure.dpi"] = 600
	#prediction_tbl = pd.DataFrame({'Likelihood': 100*prediction[0, :]}, index = index)
	#st.write(prediction_tbl)
	#ax = prediction_tbl.plot(kind='bar', figsize= (5, 7), fontsize=13, legend = False)
	#ax.set_title('Likelihood of each state')
	#ax.set_ylabel('Probability in %')
	#ax.set_xlabel('Affective State')
	plt.tight_layout()
	st.pyplot(use_container_width=True)

	
	st.subheader('Please select your desired time using the slider below')
	
	pick_time = st.slider(label = 'Minutes', min_value = 1, max_value = 40, step = 1, format = '%d')
	
	'The barplot shows the dominant state at the selected time'
	
	fig_bar, ax2 = plt.subplots(figsize = (20,15), dpi = 600)	 
	
	if pick_time <= 4: 
		 
		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[0, :], palette = colors, alpha=0.3)
	
	elif (pick_time > 4 and pick_time <= 8): 

		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[1, :], palette = colors, alpha=0.3)
	
	elif pick_time > 8 and pick_time <= 12: 
	
		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[2, :], palette = colors, alpha=0.3)	
	
	elif pick_time > 12 and pick_time <= 16: 
 
		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[3, :], palette = colors, alpha=0.3)
		
	elif pick_time > 16 and pick_time <= 20: 

		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[4, :], palette = colors, alpha=0.3)	
		
	elif pick_time > 20 and pick_time <= 24: 

		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[5, :], palette = colors, alpha=0.3)

	elif pick_time > 24 and pick_time <= 28: 
	
		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[6, :], palette = colors, alpha=0.3)	
		
	elif pick_time > 28 and pick_time <= 32: 

		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[7, :], palette = colors, alpha=0.3)
	
	else: 
		ax2 = sns.barplot(x = df1.columns, y = df1.iloc[-1, :], palette = colors, alpha=0.3)	
		
	ax2.set_ylabel("Prediction confidence %", fontsize=34)
	#ax2.legend(bbox_to_anchor=(1,1.02), fontsize=34)
	ax2.tick_params(axis='both', which='major', labelsize=34)
	ax2.tick_params(axis='both', which='minor', labelsize=24)
	ax2.tick_params(axis='x', which='major', rotation=90)
	fig_bar.tight_layout()	
	st.pyplot(fig_bar, use_container_width=True)	
		
		
		
		
		
		
	
	
