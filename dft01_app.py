#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import re
from IPython.display import display
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import shutil
import posixpath
import seaborn as sns

# EDA Pkgs
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg') 
import joblib


# In[2]:


import sys
import os
import keras as ks
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys, random
from sklearn.model_selection import train_test_split

import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import json
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from scipy.ndimage import label
import warnings
from sklearn.preprocessing import MaxAbsScaler
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, auc, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import pickle
warnings.filterwarnings("ignore")


font = {'weight' : 'bold',
        'size'   : 18}
import matplotlib
matplotlib.rc('font', **font)

from numpy import array
import keras 
from tensorflow.python.keras.utils.data_utils import Sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import Embedding
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[3]:

st.title('Welcome to StressWatch')


@st.cache
def ReadSignals():
    acc_eda_temp_dic = {}
    files = [file for file in glob.glob('stress_data/Acc_Temp_EDA/*.csv')]
    column_labels = ["Hr", "Min", "Sec", "Accz", "Accy", "Accx", "Temp", "EDA", "Label"]
    #subjects = set(file.split('/')[-1].split('Acc'))
    for subject in tqdm(files):
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
    for subject in tqdm(files2):
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


# In[4]:

# Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading data...')
# Load the participants signals into the dictionary.
signals = ReadSignals()
# Notify the reader that the data was successfully loaded.
#data_load_state.text("Done! (using st.cache)")


activities = ['EDA','Prediction']
choices = st.sidebar.selectbox("Select Activity",activities)

if choices == 'EDA':
	st.subheader("EDA")
	user = st.sidebar.selectbox('Participant choices', list(signals.keys()), 0)

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

if choices == 'Prediction':
	#st.subheader("Likelihood of being in each state")
	user = st.sidebar.selectbox('Participant choices', list(signals.keys()), 0)
	def create_data(user, N_samples):
		length = signals[user][0]['EDA'].shape[0]
		max_interval = length//N_samples
		#for i in range(max_interval): 
		#i = np.random.choice(max_interval - 1, 1, replace=True)[0]
		
		
		x = [np.hstack(signals[user][0]['AccZ'][(max_interval-1)*N_samples:(max_interval)*N_samples]), 
		    np.hstack(signals[user][0]['AccY'][(max_interval-1)*N_samples:(max_interval)*N_samples]),
		    np.hstack(signals[user][0]['AccX'][(max_interval-1)*N_samples:(max_interval)*N_samples]),
		    np.hstack(signals[user][0]['Temp'][(max_interval-1)*N_samples:(max_interval)*N_samples]),
		    np.hstack(signals[user][0]['EDA'][(max_interval-1)*N_samples:(max_interval)*N_samples]),
		    np.hstack(signals[user][1]['HeartRate'][(max_interval-1)*N_samples:(max_interval)*N_samples]),
		    np.hstack(signals[user][1]['SpO2'][(max_interval-1)*N_samples:(max_interval)*N_samples])]

		y = np.vstack(signals[user][0]['Label'][(max_interval-1)*N_samples:(max_interval)*N_samples])

		return x, y[-1]

	def batch_generator(batch_size, N_samples):
		while True:

			batch = np.array([create_data(user, N_samples)])

			batch_X = np.array([batch[i][0] for i in range(len(batch))])
			batch_Y = np.vstack([batch[i][1] for i in range(len(batch))])
			
			# yield batch_X, batch_Y
			return batch_X, batch_Y

	load_prediction_models = open("lstm_model.pkl", 'rb') 
	predictor = joblib.load(load_prediction_models)
	load_prediction_models.close()
	
	x, y = batch_generator(batch_size = 8, N_samples=240)
	prediction = predictor.predict(x)
	index = ['Cognitive Stress', 'Emotional Stress', 'Physical Stress', 'Relax']
	prediction_tbl = pd.DataFrame({'Likelihood': 100*prediction[0, :]}, index = index)
	#st.write(prediction_tbl)
	ax = prediction_tbl.plot(kind='bar', figsize= (5, 7), fontsize=13, legend = False)
	ax.set_title('Likelihood of each state')
	ax.set_ylabel('Probability in %')
	#ax.set_xlabel('Affective State')
	plt.tight_layout()
	st.pyplot()
