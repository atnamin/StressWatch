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

st.title('StressWatch')

@st.cache(persist=True)
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
data_load_state = st.text('Loading data...')
# Load the participants signals into the dictionary.
signals = ReadSignals()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")


participant = st.selectbox('Participant choices', list(signals.keys()), 0)


st.line_chart(signals[participant][0]['EDA'])
st.line_chart(signals[participant][0]['Temp'])
st.line_chart(signals[participant][1]['HeartRate'])
st.line_chart(signals[participant][1]['SpO2'])
st.line_chart(signals[participant][0]['AccZ'])
st.line_chart(signals[participant][0]['AccY'])
st.line_chart(signals[participant][0]['AccX'])

# In[6]:


class data_generator:
 
##This class has essential functions supporting fast and effective computation 
#for reading the raw data, droping the irrelevant lables, 
#preparing the input data and generating the batch to be used in NN model.
       
    
    def __init__(self, path_to_data = 'None'):
        super().__init__()
        self.path = path_to_data
        self.signals = self.ReadSignals()
        self.users = list(self.signals.keys())
        self.train = [key for key in self.users if key not in ['Subject11', 'Subject10', 'Subject4']]
        self.test = {your_key: self.signals[your_key] for your_key in ['Subject11', 'Subject10', 'Subject4']}
        self.signals = { your_key: self.signals[your_key] for your_key in self.train }
        self.users = list(self.signals.keys())
#        self.keys = ['AccZ', 'AccY', 'AccX', 'Temp', 'EDA', 'HeartRate', 'SpO2']
#        self.mask_data()
#        self.mask_test_data()
        
    
    def ReadSignals(self):
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
            if key in rsmpl_reindx_dic: signal_data.setdefault(key, []).append(rsmpl_reindx_dic[key])
            if key in hr_o2_dic: signal_data.setdefault(key, []).append(hr_o2_dic[key])


        return signal_data

    
    
    def create_data(self, user, N_samples):
#	This function creates and stacks the time series data for the train set.
#        USAGE: Create the train dataset
#        ARGS: 
#        @user = string (patient ID), 
#        @N_samples = Numeric represnting the number of samples for picking the records
#       OUTPUT: x and y as features and labels
        length = self.signals[user][0]['EDA'].shape[0]
        max_interval = length//N_samples
        i = np.random.choice(max_interval - 1, 1, replace=True)[0]
        
        x = [np.hstack(self.signals[user][0]['AccZ'][i*N_samples:(i+1)*N_samples]), 
            np.hstack(self.signals[user][0]['AccY'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.signals[user][0]['AccX'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.signals[user][0]['Temp'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.signals[user][0]['EDA'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.signals[user][1]['HeartRate'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.signals[user][1]['SpO2'][i*N_samples:(i+1)*N_samples])]

        y = self.signals[user][0]['Label'][i*N_samples:(i+1)*N_samples]
        
        return x, y[int((N_samples/2) + i*N_samples)]   
        
    
    def create_test_data(self, user, N_samples):
#	This function creates and stacks the time series data for the test set.
#        USAGE: Create the test dataset
#        ARGS: user = string (patient ID), N_samples = Numeric represnting the number of samples #		for picking the records
#        OUTPUT: X and Y as features and labels
        length = self.test[user][0]['EDA'].shape[0]
        max_interval = length//N_samples
        i = np.random.choice(max_interval - 1, 1, replace=True)[0]

        x = [np.hstack(self.test[user][0]['AccZ'][i*N_samples:(i+1)*N_samples]), 
            np.hstack(self.test[user][0]['AccY'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.test[user][0]['AccX'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.test[user][0]['Temp'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.test[user][0]['EDA'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.test[user][1]['HeartRate'][i*N_samples:(i+1)*N_samples]),
            np.hstack(self.test[user][1]['SpO2'][i*N_samples:(i+1)*N_samples])]

        y = self.test[user][0]['Label'][i*N_samples:(i+1)*N_samples]
            
        return x, y[int((N_samples/2) + i*N_samples)]


    def batch_generator_train(self, batch_size, N_samples):
#	This function generates the batch for the train set.
#        USAGE: Generate the train batch 
#        ARGS: batch_size = Numeric representing the batch size (number of patients), N_samples  #		Numeric represnting the number of samples for picking the records
#        OUTPUT: Train batch for X and Y
        while True:
            # create the indicies
            self.batch_indices_tr = np.random.choice(len(self.users), batch_size, replace=True)  
            users_to_pick = [self.users[i] for i in self.batch_indices_tr]
            
            batch_tr = np.array([self.create_data(user, N_samples) for user in users_to_pick])
            
            batch_tr_X = np.array([batch_tr[i][0] for i in range(len(batch_tr))])
            batch_tr_Y = np.vstack([batch_tr[i][1] for i in range(len(batch_tr))])
            # yield the data
            
            yield batch_tr_X, batch_tr_Y
                        
    def batch_generator_validation(self, batch_size , N_samples):
#	This function generates the batch for the validation set.
#        USAGE: Generate the validation batch 
#        ARGS: batch_size = Numeric representing the batch size (number of patients), N_samples #	= Numeric represnting the number of samples for picking the records
#        OUTPUT: Validation batch for X and Y
        while True:
            # create the indicies
            self.batch_indices_tr = np.random.choice(len(self.users), batch_size, replace=True)  
            users_to_pick = [self.users[i] for i in self.batch_indices_tr]
            
            batch_tr = np.array([self.create_data(user, N_samples) for user in users_to_pick])
            
            batch_tr_X = np.array([batch_tr[i][0] for i in range(len(batch_tr))])
            batch_tr_Y = np.vstack([batch_tr[i][1] for i in range(len(batch_tr))])
            
            # yield the data
            yield batch_tr_X, batch_tr_Y
    
    
    def batch_generator_test(self, batch_size , N_samples):
#	This function generates the batch for the test set.
#        USAGE: Generate the test batch 
#        ARGS: batch_size = Numeric representing the batch size (number of patients), N_samples #	= Numeric represnting the number of samples for picking the records
#        OUTPUT: Test batch for X and Y
       while True:

            batch_test = np.array([self.create_test_data(user, N_samples) for user in data_gen.test.keys()])
            
            batch_test_X = np.array([batch_test[i][0] for i in range(len(batch_test))])
            batch_test_Y = np.vstack([batch_test[i][1] for i in range(len(batch_test))])
            
            # yield the data
            yield batch_test_X, batch_test_Y
#             batch_test_X = np.array([batch_test[i][0] for i in range(len(batch_test))])
#             batch_test_Y = np.vstack([batch_test[i][1] for i in range(len(batch_test))])
            # yield the data

            #return batch_test
     


# In[77]:


data_gen = data_generator()


# In[58]:


x, y = next(data_gen.batch_generator_train(batch_size=4, N_samples=300))
n_timesteps, n_features, n_outputs = x.shape[1], x.shape[2], y.shape[1]


# In[70]:


model = Sequential()
model.add(LSTM(300, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[71]:


model.summary()


# In[10]:


early_stopping_callback = EarlyStopping(monitor='val_loss', patience=70, verbose = 0, mode = 'min')
checkpoint_callback = ModelCheckpoint('keras_checkpoint_1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[11]:


history = model.fit_generator(data_gen.batch_generator_train(batch_size=4, N_samples=300), steps_per_epoch = 32, 
                    verbose=1, validation_data=data_gen.batch_generator_validation(batch_size=2, N_samples=300), validation_steps = 64,
                   epochs=20, callbacks=[early_stopping_callback, checkpoint_callback])


# In[22]:


loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,21)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

st.pyplot()
# In[24]:


acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1,21)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

st.pyplot()
# In[83]:





