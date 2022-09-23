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

file = open('MyoArmband/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
fs = 200
w_ms = 500 #Largo de la ventana
s_ms = 50 #Incremento
W = int(fs*w_ms/1000)
I = int(fs*s_ms/1000)
data = MyoArm_data['S3']['emg']
etiqueta = MyoArm_data['S3']['label']

# Normalización
for j in range(16):
    data[:,j] = data[:,j]/max(abs(data[:,j]))
    
for i in range(2, 2):
    S_aux = MyoArm_data['S'+str(i)]['emg']
    label_aux = MyoArm_data['S'+str(i)]['label']
    # Normalizacion
    for j in range(16):
        # print(max(np.max(S_aux[:,j]), abs(np.min(S_aux[:,j]))))
        S_aux[:,j] = S_aux[:,j]/max(abs(S_aux[:,j]))
    etiqueta = np.concatenate((etiqueta, label_aux))
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
        mav[i] = features.mav(data[:,j][i*I:(i)*I+W])
        ls[i] = features.ls(data[:,j][i*I:(i)*I+W])
        mfl[i] = features.mfl(data[:,j][i*I:(i)*I+W])
        msr[i] = features.msr(data[:,j][i*I:(i)*I+W])
        wamp[i] = features.wamp(data[:,j][i*I:(i)*I+W])
        iav[i] = features.iav(data[:,j][i*I:(i)*I+W])
        dasdv[i] = features.dasdv(data[:,j][i*I:(i)*I+W])
        _var[i] = features._var(data[:,j][i*I:(i)*I+W])
        rms[i] = features.rms(data[:,j][i*I:(i)*I+W])
        DWT1[i],DWT2[i],DWT3[i] = features.DWT(data[:,j][i*I:(i)*I+W])
    FEATURES["wl_c"+str(j+1)] = wl/max(abs(wl))
    FEATURES["zc_c"+str(j+1)] = zc/max(abs(zc))  
    FEATURES["ssc_c"+str(j+1)] = ssc/max(abs(ssc))  
    # FEATURES["rms_c"+str(j+1)] = rms/max(abs(rms))  
    # FEATURES["mav_c"+str(j+1)] = mav/max(abs(mav))
    # FEATURES["ls_c"+str(j+1)] = ls/max(abs(ls))
    # FEATURES["mfl_c"+str(j+1)] = mfl/max(abs(mfl))  
    # FEATURES["msr_c"+str(j+1)] = msr/max(abs(msr))
    # FEATURES["wamp_c"+str(j+1)] = wamp/max(abs(min(wamp)), max(wamp))  
    # FEATURES["iav_c"+str(j+1)] = iav/max(abs(iav))
    # FEATURES["dasdv_c"+str(j+1)] = dasdv/max(abs(dasdv)) 
    # FEATURES["var_c"+str(j+1)] = _var/max(abs(_var)) 
    FEATURES["DWT1_c"+str(j+1)] = DWT1/max(abs(DWT1)) 
    FEATURES["DWT2_c"+str(j+1)] = DWT2/max(abs(DWT2)) 
    FEATURES["DWT3_c"+str(j+1)] = DWT3/max(abs(DWT3)) 
label = np.zeros(l)
for i in range(l):
    if((etiqueta[i*I:(i)*I+W]>0).sum()>I/2):
        label[i] = max(etiqueta[i*I:(i)*I+W])
    else:
        label[i] = 0
label = pd.Series(label, dtype="category")



# Agrupación de movimientos
sub_cat_1 = [20]
cat_1 = 52
sub_cat_2 = [21,22,25,26,27,28]
cat_2 = 20
sub_cat_3 = [30, 31,32,34,36,37,39,40,41,42]
cat_3 = 21 #Debe ir despues de asignar la cat_1 etiqueta 21 es "ficticia" equivale a la 30
sub_cat_4 = [33]
cat_4 = 19

new_label = features.re_label(label, sub_cat_1, cat_1)
new_label = features.re_label(new_label, sub_cat_2, cat_2)
new_label = features.re_label(new_label, sub_cat_3, cat_3)
new_label = features.re_label(new_label, sub_cat_4, cat_4)

dump(FEATURES, "FEATURES.joblib")
dump(new_label, "label.joblib")