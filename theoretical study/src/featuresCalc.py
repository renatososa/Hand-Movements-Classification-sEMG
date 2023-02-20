# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:13:18 2022
This script corresponds to the calculation of the features for the training of the algorithms.
@Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""

import numpy as np
import pickle
from joblib import dump
import pandas as pd
import sys

sys.path.append('../lib')
import functions as f

# Load data
file = open('../data/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()

# Set parameters
fs = MyoArm_data['S1']['info']['fs'] # Frecuency in Hz
nOfSubjects = 2 # N° of Subjects [2,...,10]
nChannels = 16 # N° of Channels
w_ms = 1000 # Window length in ms
s_ms = 50 # Window increment in ms
W_s = int(fs*w_ms/1000) # Window length in samples
I_s = int(fs*s_ms/1000) # Window increment in samples
cor = 200 # N° of samples in label correction

# Data extraction
data = MyoArm_data['S1']['emg']
label = MyoArm_data['S1']['label']
suject = label*0 + 1
# Normalization
for j in range(nChannels):
    data[:,j] = data[:,j]/np.std(data[:,j])-np.mean(data[:,j])
    
for i in range(2, nOfSubjects+1):
    # Data extraction
    S_aux = MyoArm_data['S'+str(i)]['emg']
    label_aux = MyoArm_data['S'+str(i)]['label']
    sujeto_aux = label_aux*0 + i
    # Normalization
    for j in range(nChannels):
        S_aux[:,j] = S_aux[:,j]//np.std(S_aux[:,j])-np.mean(S_aux[:,j])
    label = np.concatenate((label, label_aux))
    suject = np.concatenate((suject, sujeto_aux))
    data = np.concatenate((data, S_aux))

# Features calculation
l = int((len(data)-W_s+I_s)/I_s) # N° of windows in the register
zcThreshold = 0.05
sscThreshold = 0.05
rms = np.zeros(l)
zc = np.zeros(l)
ssc = np.zeros(l)
wl = np.zeros(l)
mav = np.zeros(l)
ls = np.zeros(l)
mfl = np.zeros(l)
msr = np.zeros(l)
wamp = np.zeros(l)
iav = np.zeros(l)
dasdv = np.zeros(l)
_var = np.zeros(l)
DWT1 = np.zeros(l)
DWT2 = np.zeros(l)
DWT3 = np.zeros(l)

FEATURES = pd.DataFrame()
for j in range(nChannels):
    for i in range(l):
        wl[i] = f.wl(data[:,j][i*I_s:(i+1)*I_s+W_s])
        zc[i] = f.zc(data[:,j][i*I_s:(i+1)*I_s+W_s], zcThreshold)
        ssc[i] = f.ssc(data[:,j][i*I_s:(i+1)*I_s+W_s], sscThreshold)
        rms[i] = f.rms(data[:,j][i*I_s:(i+1)*I_s+W_s])
        mav[i] = f.mav(data[:,j][i*I_s:(i+1)*I_s+W_s])
        ls[i] = f.ls(data[:,j][i*I_s:(i+1)*I_s+W_s])
        mfl[i] = f.mfl(data[:,j][i*I_s:(i+1)*I_s+W_s])
        msr[i] = f.msr(data[:,j][i*I_s:(i+1)*I_s+W_s])
        wamp[i] = f.wamp(data[:,j][i*I_s:(i+1)*I_s+W_s])
        iav[i] = f.iav(data[:,j][i*I_s:(i+1)*I_s+W_s])
        dasdv[i] = f.dasdv(data[:,j][i*I_s:(i+1)*I_s+W_s])
        _var[i] = f._var(data[:,j][i*I_s:(i+1)*I_s+W_s])        
        DWT1[i],DWT2[i],DWT3[i] = f.DWT(data[:,j][i*I_s:(i+1)*I_s+W_s])
    FEATURES["wl_c"+str(j+1)] = wl/max(abs(wl))
    FEATURES["zc_c"+str(j+1)] = zc/max(abs(zc))  
    FEATURES["ssc_c"+str(j+1)] = ssc/max(abs(ssc))  
    FEATURES["rms_c"+str(j+1)] = rms/max(abs(rms))  
    FEATURES["mav_c"+str(j+1)] = mav/max(abs(mav))
    FEATURES["ls_c"+str(j+1)] = ls/max(abs(ls))
    FEATURES["mfl_c"+str(j+1)] = mfl/max(abs(mfl))  
    FEATURES["msr_c"+str(j+1)] = msr/max(abs(msr))
    FEATURES["wamp_c"+str(j+1)] = wamp/max(abs(min(wamp)), max(wamp))  
    FEATURES["iav_c"+str(j+1)] = iav/max(abs(iav))
    FEATURES["dasdv_c"+str(j+1)] = dasdv/max(abs(dasdv)) 
    FEATURES["var_c"+str(j+1)] = _var/max(abs(_var)) 
    FEATURES["DWT1_c"+str(j+1)] = DWT1/max(abs(DWT1)) 
    FEATURES["DWT2_c"+str(j+1)] = DWT2/max(abs(DWT2)) 
    FEATURES["DWT3_c"+str(j+1)] = DWT3/max(abs(DWT3)) 

# Label correction
events = np.where(np.diff(label.T)[0]!=0)[0]
nOfEvents = int(len(events)/2)
for i in range(nOfEvents):
    label[events[i*2]:events[i*2]+cor] = 0
    label[events[i*2+1]-cor:events[i*2+1]+1] = 0

# Window label assignment (the most frequent)
subLabel = np.zeros(l)
subSubject = np.zeros(l)
for i in range(l):
    if((label[i*I_s:(i)*I_s+W_s]>0).sum()>I_s/2):
        subLabel[i] = max(label[i*I_s:(i)*I_s+W_s])
    else:
        subLabel[i] = 0
    subSubject[i] = suject[i*I_s]

# Data exportation
labels = {'mov':subLabel, 'subjet': subSubject}
labels = pd.DataFrame(labels)
dump(FEATURES, "../data/FEATURES.joblib")
dump(labels, "../data/label.joblib")