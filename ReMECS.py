#!/usr/bin/env python
# coding: utf-8

# ## ReMECS -- Real-time Multimodal Emotion Classification System

# In[8]:
#============================
# Import important libraries
#============================
import pandas as pd 
import numpy as np
import math
import scipy
import pywt
from river import metrics
import time
import datetime
from statistics import mode
from scipy import stats
from sklearn import preprocessing
from collections import defaultdict, Counter
from scipy.special import expit
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

from ANNCustom import ANNCustom
from window_slider import Slider
import warnings
warnings.filterwarnings("ignore")


print('ReMECS started!!')



# In[10]:


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values)> 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):    
    list_values = list_values[0,:]
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


# In[11]:


#======================================================
# EDA Feature Extraction (Wavelet Features)
#======================================================
def extract_eda_features(raw_eda):
    features =[]
    EDA = raw_eda
    list_coeff = pywt.wavedec(EDA, 'db4', level=3)
    
#     print(list_coeff)
    for coeff in list_coeff:
        features += get_features(coeff)
    return features


# In[12]:


#======================================================
# RESP BELT Feature Extraction (Wavelet Features)
#======================================================

def extract_resp_belt_features(raw_data):
    features =[]
    resp_belt = raw_data
    list_coeff = pywt.wavedec(resp_belt, 'db4', level=3)
    
#     print(list_coeff)
    for coeff in list_coeff:
        features += get_features(coeff)
    return features


# In[13]:


def eeg_features(raw_data):
    ch = 0
    features= []
    def calculate_entropy(list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1]/len(list_values) for elem in counter_values]
        entropy=scipy.stats.entropy(probabilities)
        return entropy

    def calculate_statistics(list_values):
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values**2))
        return [median, mean, std, var, rms]

    def get_features(list_values):    
    #     list_values = list_values[0,:]
        entropy = calculate_entropy(list_values)
        statistics = calculate_statistics(list_values)
        return [entropy] + statistics
    
    for i in range(raw_data.shape[0]):
        ch_data = raw_data[i]
        list_coeff = pywt.wavedec(ch_data, 'db4', level=5)
        for coeff in list_coeff:
            features += get_features(coeff)
            
        ch = ch+1
    return features


# In[14]:


##===================================================
# EEG data read from files
##===================================================
def eeg_data(p,v):
    file_eeg = 'data/eeg_data/'+str(p)+'_data_DEAP'+'.csv'
#     print(file_eeg)
    df = pd.read_csv(file_eeg,sep=',', header = None, engine='python')
    eeg_sig = df.loc[df.iloc[:,1] == v]
    return eeg_sig

##===================================================
# EDA data read from files
##===================================================
def eda_data(p,v):
    file_eda = 'data/eda_data/'+str(p)+'_GSR_data_from_DEAP.csv'
#     print(file_eda)
    df = pd.read_csv(file_eda,sep=',', header = None, engine='python')
    eda_sig = df.loc[df.iloc[:,1] == v]
    return eda_sig

##===================================================
# Resp data read from files
##===================================================
def resp_data(p,v):
    file_resp = 'data/resp_data/'+str(p)+'_Respiration_data_from_DEAP.csv'
#     print(file_resp)
    df = pd.read_csv(file_resp,sep=',', header = None, engine='python')
    resp_sig = df.loc[df.iloc[:,1] == v]
    return resp_sig


# In[15]:


#=======================================
# MAIN PROGRAM STARTS HERE
#=======================================

segment_in_sec = 10 #in sec
bucket_size = int((8064/60)*segment_in_sec)  #8064 is for 60 sec record
overlap_count = 0

num_classifiers = 3 #Total number of classifiers
w_val =np.ones(num_classifiers)/num_classifiers #Weights for valence classifiers
w_aro =np.ones(num_classifiers)/num_classifiers #Weights for valence classifiers
beta = 0.5

# l_max =0.9999
# lr_min = 0.0001
lr = 0.05
b =1 

epochs =  1 #epoch is 1 because the model will be trained only once


optimizer= 'SGD' #optimizer
classifier = 'MLP_'+str(optimizer)
run = 1

participant = 32
videos = 40

global eeg_emotion, eda_emotion, resp_emotion, mer_emotion, all_eta
eeg_emotion = []
eda_emotion = []
resp_emotion = []
mer_emotion = []



all_eta =[]
init_m = 0

#================================================
# Performance matric declaration here
#================================================
mer_acc_val = metrics.Accuracy() #Accuracy
mer_f1m_val = metrics.F1() #F1 measure  
mer_acc_aro = metrics.Accuracy() #Accuracy
mer_f1m_aro = metrics.F1() #F1 measure



eeg_acc_val = metrics.Accuracy() #Accuracy
eeg_f1m_val = metrics.F1() #F1 measure  
eeg_acc_aro = metrics.Accuracy() #Accuracy
eeg_f1m_aro = metrics.F1() #F1 measure


eda_acc_val = metrics.Accuracy() #Accuracy
eda_f1m_val = metrics.F1() #F1 measure  
eda_acc_aro = metrics.Accuracy() #Accuracy
eda_f1m_aro = metrics.F1() #F1 measure


resp_acc_val = metrics.Accuracy() #Accuracy
resp_f1m_val = metrics.F1() #F1 measure  
resp_acc_aro = metrics.Accuracy() #Accuracy
resp_f1m_aro = metrics.F1() #F1 measure


 
mer_cm_val  = metrics.ConfusionMatrix()
mer_cm_aro  = metrics.ConfusionMatrix()

eeg_cm_val  = metrics.ConfusionMatrix()
eeg_cm_aro  = metrics.ConfusionMatrix()

eda_cm_val  = metrics.ConfusionMatrix()
eda_cm_aro  = metrics.ConfusionMatrix()
  
resp_cm_val  = metrics.ConfusionMatrix()
resp_cm_aro  = metrics.ConfusionMatrix()
    
itr = 0 #controls the learning rate


for ii in range(0,participant):


        p =ii+1
        for jj in range(0,videos):
            v = jj+1
            print('===============================================================')
            p_v = 'Person:'+ ' ' +str(p)+ ' ' +'Video:'+str(v)
            print(p_v)
            
            print('------------------------------------------------')
            ##===================================================
            # Data read from files
            ##===================================================
            eeg_sig = eeg_data(p,v)
            eda_sig = eda_data(p,v)
            resp_sig = resp_data(p,v)
            
            #=================================================
            #emotion labels (valence, arousal) mapping 0-1
            #=================================================
            val = eeg_sig.iloc[0,8067]
            aro = eeg_sig.iloc[0,8068]
            
            #valence emotion maping 0-> low valence and 1-> high valence

            if (val >5):
                vl = 1 #high valence
            else:
                vl = 0 #low valence

            #arousal emotion maping 0-> low arousal and 1-> high high arousal
            if (aro >5):
                al = 1 #high arousal
            else:
                al = 0 #low arousal
                
            y_act_val = np.array([[vl]])
            y_act_aro = np.array([[al]]) 
            
            
            #=========================================
            # Sliding window starts here 
            #=========================================
            slider_eeg = Slider(bucket_size,overlap_count)
            slider_eda = Slider(bucket_size,overlap_count)
            slider_resp = Slider(bucket_size,overlap_count)
            
            eeg_sig = np.array(eeg_sig.iloc[range(0,32),range(3,8067)]) #keeping only eeg signals
            eda_sig = np.array(eda_sig.iloc[:,range(3,8067)]) #keeping only eda signals
            resp_sig = np.array(resp_sig.iloc[:,range(3,8067)]) #keeping only resp signals
            
            slider_eeg.fit(eeg_sig)
            slider_eda.fit(eda_sig)
            slider_resp.fit(resp_sig)

            while True:
                window_data_eeg = slider_eeg.slide()
                window_data_eda = slider_eda.slide() 
                window_data_resp = slider_resp.slide() 
                
                #=================================================
                # Feature extraction from EEG
                #=================================================
                features_eeg = eeg_features(window_data_eeg)
                x_eeg = np.array([features_eeg])  #EEG raw feature vector
                x_eeg = preprocessing.normalize(x_eeg) # EEG normalized features [0,1] 
                
                
                #=================================================
                # Feature extraction from EDA
                #=================================================
                eda_features = extract_eda_features(np.array(window_data_eda))
                x_eda = np.array([eda_features]) #EDA raw feature vector
                x_eda = preprocessing.normalize(x_eda) #EDA normalized features
                
                #=================================================
                # Feature extraction from Resp belt
                #=================================================

                resp_features = extract_resp_belt_features(np.array(window_data_resp))
                x_resp = np.array([resp_features]) #RESP BELT raw feature vector
                x_resp = preprocessing.normalize(x_resp) #RESP BELT normalized features
            
            
                #===================================================
                # Model initialization
                #===================================================
                if init_m == 0:
                    print('EEG Feature shape{}:'.format(x_eeg.shape))
                    print('EDA Feature shape{}:'.format(x_eda.shape))
                    print('RESP BELT Feature shape{}:'.format(x_resp.shape))

                    eeg_size_hidden = 30 #math.ceil((2/3)*x_eeg.shape[1]) #Hidden node size
                    eda_size_hidden = math.ceil((2/3)*x_eda.shape[1]) #Hidden node size
                    resp_size_hidden = math.ceil((2/3)*x_resp.shape[1]) #Hidden node size
                    
                    output_class = y_act_val.shape[0]

                    #========================
                    # For EEG data MLP model
                    #========================
                    eeg_model_val = ANNCustom()
                    eeg_model_val.add_layer(x_eeg.shape[1], eeg_size_hidden, 'sigmoid')
                    eeg_model_val.add_layer(eeg_size_hidden, output_class, 'sigmoid')
                    eeg_model_val.compile(learning_rate=lr)
                    


                    eeg_model_aro = ANNCustom()
                    eeg_model_aro.add_layer(x_eeg.shape[1], eeg_size_hidden, 'sigmoid')
                    eeg_model_aro.add_layer(eeg_size_hidden, output_class, 'sigmoid')
                    eeg_model_aro.compile(learning_rate=lr)
                    

                    #========================
                    # For EDA data MLp model
                    #========================
                    eda_model_val = ANNCustom()
                    eda_model_val.add_layer(x_eda.shape[1], eda_size_hidden, 'sigmoid')
                    eda_model_val.add_layer(eda_size_hidden, output_class, 'sigmoid')
                    eda_model_val.compile(learning_rate=lr)
                    

                    eda_model_aro = ANNCustom()
                    eda_model_aro.add_layer(x_eda.shape[1], eda_size_hidden, 'sigmoid')
                    eda_model_aro.add_layer(eda_size_hidden, output_class, 'sigmoid')
                    eda_model_aro.compile(learning_rate=lr)
                    

                    #==============================
                    # For Resp Belt data MLP Model
                    #==============================
                    resp_model_val = ANNCustom()
                    resp_model_val.add_layer(x_resp.shape[1], resp_size_hidden, 'sigmoid')
                    resp_model_val.add_layer(resp_size_hidden, output_class, 'sigmoid')
                    resp_model_val.compile(learning_rate=lr)
                    

                    resp_model_aro = ANNCustom()
                    resp_model_aro.add_layer(x_eda.shape[1], resp_size_hidden, 'sigmoid')
                    resp_model_aro.add_layer(resp_size_hidden, output_class, 'sigmoid')
                    resp_model_aro.compile(learning_rate=lr)
                    
                    init_m = init_m+1


                #===============================================================
                # Emotion Classification --> Valence and Arousal
                #===============================================================

                #===========================================
                # From EEG data -- RECS System
                #===========================================

                #Valence classification EEG

                #Test the model first 
                y_pred_val_eeg = eeg_model_val.predict_once(x_eeg)

                #Train the model once
                eeg_model_val.fit_once(x_eeg,y_act_val,epochs=epochs)

                eeg_acc_val.update(y_act_val[0][0], y_pred_val_eeg)  # update the accuracy metric

                eeg_f1m_val.update(y_act_val[0][0], y_pred_val_eeg) #update f1 measure metric
                
            

                #Arousal classification EEG

                #Test the model first 
                y_pred_aro_eeg = eeg_model_aro.predict_once(x_eeg)

                #Train the model once
                eeg_model_aro.fit_once(x_eeg,y_act_aro,epochs=epochs)

                eeg_acc_aro.update(y_act_aro[0][0], y_pred_aro_eeg)  # update the accuracy metric

                eeg_f1m_aro.update(y_act_aro[0][0], y_pred_aro_eeg) #update f1 measure metric
                
                

                #===========================================
                # From EDA data 
                #===========================================

                #Valence classification EDA

                #Test the model first 
                y_pred_val_eda = eda_model_val.predict_once(x_eda)

                #Train the model once
                eda_model_val.fit_once(x_eda,y_act_val,epochs=epochs)

                eda_acc_val.update(y_act_val[0][0], y_pred_val_eda)  # update the accuracy metric

                eda_f1m_val.update(y_act_val[0][0], y_pred_val_eda) #update f1 measure metric
                
                
                

                #Arousal classification EDA
                #Test the model first 
                y_pred_aro_eda = eda_model_aro.predict_once(x_eda)

                #Train the model once
                eda_model_aro.fit_once(x_eda,y_act_aro,epochs=epochs)

                eda_acc_aro.update(y_act_aro[0][0], y_pred_aro_eda)  # update the accuracy metric

                eda_f1m_aro.update(y_act_aro[0][0], y_pred_aro_eda) #update f1 measure metric
                
                

                #===========================================
                # From Resp Belt data
                #===========================================

                #Valence classification Resp Belt

                #Test the model first 
                y_pred_val_resp = resp_model_val.predict_once(x_resp)

                #Train the model once
                resp_model_val.fit_once(x_resp,y_act_val,epochs=epochs)

                resp_acc_val.update(y_act_val[0][0], y_pred_val_resp)  # update the accuracy metric

                resp_f1m_val.update(y_act_val[0][0], y_pred_val_resp) #update f1 measure metric
                
                
                

                #Arousal classification Resp Belt
                #Test the model first 
                y_pred_aro_resp = resp_model_aro.predict_once(x_resp)

                #Train the model once
                resp_model_aro.fit_once(x_resp,y_act_aro,epochs=epochs)

                resp_acc_aro.update(y_act_aro[0][0], y_pred_aro_resp)  # update the accuracy metric

                resp_f1m_aro.update(y_act_aro[0][0], y_pred_aro_resp) #update f1 measure metric
                
                
                
                ##=============================
                # Confusion Matric Calculation
                ##=============================

                eeg_cm_val.update(y_act_val[0][0], y_pred_val_eeg)
                eeg_cm_aro.update(y_act_aro[0][0], y_pred_aro_eeg)

                eda_cm_val.update(y_act_val[0][0], y_pred_val_eda)
                eda_cm_aro.update(y_act_aro[0][0], y_pred_aro_eda)

                resp_cm_val.update(y_act_val[0][0], y_pred_val_resp)
                resp_cm_aro.update(y_act_aro[0][0], y_pred_aro_resp)            

                #=============================================================
                # Storing All Results
                #=============================================================
                
                emotion_label =[]

                emotion_label.append([y_pred_val_eeg,y_pred_aro_eeg]) #appending valence & arousal predicted EEG
                emotion_label.append([y_pred_val_eda, y_pred_aro_eda]) #appending valence & arousal predicted EDA
                emotion_label.append([y_pred_val_resp, y_pred_aro_resp]) #appending valence & arousal predicted resp

                emotion_label = np.array(emotion_label)


                #==============================================================
                # Decision label ensemble --> Weighted Majority Voting
                #==============================================================
                val_label = emotion_label[:,0]
                aro_label = emotion_label[:,1]

                #------------------------------------------
                # Valence Class ensemble
                #------------------------------------------

                p_val = np.dot([w_val],val_label)
                y_prdt_mer_val = (p_val -0.5)
                if(y_prdt_mer_val > 0):
                    mer_val = 1
                else:
                    mer_val = 0

                for i in range(val_label.shape[0]):
                    if(val_label[i] != y_act_val):
                        w_val[i] = beta*w_val[i]

                w_val_sum = sum(w_val) #total sum of weights

                w_val = np.array(w_val/w_val_sum) #weight rescaling

                #------------------------------------------
                # Arousal Class ensemble
                #------------------------------------------            
                p_val = np.dot([w_aro],aro_label)
                y_prdt_mer_aro = (p_val-0.5)
                if(y_prdt_mer_aro > 0):
                    mer_aro = 1
                else:
                    mer_aro = 0

                for i in range(aro_label.shape[0]):
                    if(aro_label[i] != y_act_aro):
                        w_aro[i] = beta*w_aro[i]


                w_aro_sum = sum(w_aro) #total sum of weights

                w_aro = np.array(w_aro/w_aro_sum) #weight rescaling

                #========================================================
                # ReMECS performance metric 
                #========================================================
                mer_acc_val.update(y_act_val[0][0], mer_val)
                mer_f1m_val.update(y_act_val[0][0], mer_val)
                mer_acc_aro.update(y_act_aro[0][0], mer_aro)
                mer_f1m_aro.update(y_act_aro[0][0], mer_aro)
                
                mer_cm_val.update(y_act_val[0][0], mer_val)
                mer_cm_aro.update(y_act_aro[0][0], mer_aro)

                
#                 eeg_emotion.append(np.array([p,v,eeg_acc_val.get(), eeg_f1m_val.get(), 
#                                              eeg_roc_val.get(),eeg_mcc_val.get(),
#                                              eeg_acc_aro.get(), eeg_f1m_aro.get(),
#                                              eeg_roc_aro.get(), eeg_mcc_aro.get(), 
#                                              y_act_val[0][0], y_pred_val_eeg, 
#                                              y_act_aro[0][0], y_pred_aro_eeg]))
            
#                 eda_emotion.append(np.array([p,v,eda_acc_val.get(), eda_f1m_val.get(), 
#                                              eda_roc_val.get(),eda_mcc_val.get(),
#                                              eda_acc_aro.get(), eda_f1m_aro.get(), 
#                                              eda_roc_aro.get(), eda_mcc_aro.get(),
#                                              y_act_val[0][0], y_pred_val_eda, 
#                                              y_act_aro[0][0], y_pred_aro_eda]))

#                 resp_emotion.append(np.array([p,v, resp_acc_val.get(), resp_f1m_val.get(), 
#                                               resp_roc_val.get(), resp_mcc_val.get(),
#                                               resp_acc_aro.get(),resp_f1m_aro.get(), 
#                                               resp_roc_aro.get(), resp_mcc_aro.get(), 
#                                               y_act_val[0][0], y_pred_val_resp, 
#                                               y_act_aro[0][0], y_pred_aro_resp]))

#                 mer_emotion.append(np.array([p,v,mer_acc_val.get(), mer_f1m_val.get(), 
#                                              mer_roc_val.get(), mer_mcc_val.get(),
#                                              mer_acc_aro.get(), mer_f1m_aro.get(), 
#                                              mer_roc_aro.get(), mer_mcc_aro.get(), 
#                                              y_act_val[0][0], mer_val, 
#                                              y_act_aro[0][0], mer_aro]))

                

                if slider_eeg.reached_end_of_list(): break

                





