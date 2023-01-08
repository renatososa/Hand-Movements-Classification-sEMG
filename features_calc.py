# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:13:18 2022

@author: Renato
"""

import numpy as np
import pickle
import extraccion_caract as features
from joblib import dump
import pandas as pd

file = open('datos/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
fs = 200
w_ms = 500 #Largo de la ventana
s_ms = 50 #Incremento
W = int(fs*w_ms/1000)
I = int(fs*s_ms/1000)
data = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']
sujeto = etiqueta*0 + 1

# Normalización
for j in range(16):
    data[:,j] = data[:,j]/max(abs(data[:,j]))
    
for i in range(2, 2):
    S_aux = MyoArm_data['S'+str(i)]['emg']
    label_aux = MyoArm_data['S'+str(i)]['label']
    sujeto_aux = label_aux*0 +i
    # Normalizacion
    for j in range(16):
        # print(max(np.max(S_aux[:,j]), abs(np.min(S_aux[:,j]))))
        S_aux[:,j] = S_aux[:,j]/max(abs(S_aux[:,j]))
    etiqueta = np.concatenate((etiqueta, label_aux))
    sujeto = np.concatenate((sujeto, sujeto_aux))
    data = np.concatenate((data, S_aux))
    
l = int((len(data)-W+I)/I)

# Corrección de etiquetas
eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]
for i in range(int(len(eventos)/2)):
    etiqueta[eventos[i*2]:eventos[i*2]+50] = 0
    etiqueta[eventos[(i+1)*2]-50:eventos[(i+1)*2]] = 0

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


# Ver algoritmos de selección de características
FEATURES = pd.DataFrame()
for j in range(16):
    for i in range(l):
        wl[i] = features.wl(data[:,j][i*I:(i)*I+W])
        zc[i] = features.zc(data[:,j][i*I:(i)*I+W], 0.05)
        ssc[i] = features.ssc(data[:,j][i*I:(i)*I+W], 0.05)
        rms[i] = features.rms(data[:,j][i*I:(i)*I+W])
        mav[i] = features.mav(data[:,j][i*I:(i)*I+W])
        ls[i] = features.ls(data[:,j][i*I:(i)*I+W])
        mfl[i] = features.mfl(data[:,j][i*I:(i)*I+W])
        msr[i] = features.msr(data[:,j][i*I:(i)*I+W])
        wamp[i] = features.wamp(data[:,j][i*I:(i)*I+W])
        iav[i] = features.iav(data[:,j][i*I:(i)*I+W])
        dasdv[i] = features.dasdv(data[:,j][i*I:(i)*I+W])
        _var[i] = features._var(data[:,j][i*I:(i)*I+W])        
        DWT1[i],DWT2[i],DWT3[i] = features.DWT(data[:,j][i*I:(i)*I+W])
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
    
label = np.zeros(l)
subject = np.zeros(l)
for i in range(l):
    if((etiqueta[i*I:(i)*I+W]>0).sum()>I/2):
        label[i] = max(etiqueta[i*I:(i)*I+W])
    else:
        label[i] = 0
    subject[i] = sujeto[i*I]
labels = {'mov':label, 'subjet': subject}
labels = pd.DataFrame(labels)

dump(FEATURES, "datos/FEATURES.joblib")
dump(labels, "datos/label.joblib")