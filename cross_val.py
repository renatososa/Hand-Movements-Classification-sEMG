# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:20:51 2022

@author: Renato
"""

import pickle
import numpy as np
import extraccion_caract as features
import pandas as pd

# Para crear redes neuronales:
from sklearn.neural_network import MLPClassifier

# Para dividir datos en entrenamiento/test:
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

# Para medir tiempos
from time import time

file = open('MyoArmband/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
fs = 200
w_ms = 1000 #Largo de la ventana
s_ms = 50 #Incremento
W = int(200*w_ms/1000)
I = int(200*s_ms/1000)
data = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']

# Normalización
for j in range(16):
    data[:,j] = data[:,j]/max(np.max(data[:,j]), abs(np.min(data[:,j]))) 

for i in range(2, 2):
    S_aux = MyoArm_data['S'+str(i)]['emg']
    label_aux = MyoArm_data['S'+str(i)]['label']
    # Normalizacion
    for j in range(16):
        S_aux[:,j] = S_aux[:,j]/max(np.max(S_aux[:,j]), abs(np.min(S_aux[:,j])))     
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

# Ver algoritmos de selección de características
FEATURES = pd.DataFrame()
for j in range(16):
    for i in range(l):
        # rms[i] = features.rms(data[:,j][i*I:(i)*I+W])
        wl[i] = features.wl(data[:,j][i*I:(i)*I+W])
        zc[i] = features.zc(data[:,j][i*I:(i)*I+W], 0.01)
        ssc[i] = features.ssc(data[:,j][i*I:(i)*I+W], 0.01)
        # mav[i] = features.mav(data[:,j][i*I:(i)*I+W])
        # ls[i] = features.ls(data[:,j][i*I:(i)*I+W])
        # mfl[i] = features.mfl(data[:,j][i*I:(i)*I+W])
        # msr[i] = features.msr(data[:,j][i*I:(i)*I+W])
        # wamp[i] = features.wamp(data[:,j][i*I:(i)*I+W])
        # iav[i] = features.iav(data[:,j][i*I:(i)*I+W])
        # dasdv[i] = features.dasdv(data[:,j][i*I:(i)*I+W])
        # _var[i] = features._var(data[:,j][i*I:(i)*I+W])
    # FEATURES["rms_c"+str(j+1)] = rms       
    FEATURES["wl_c"+str(j+1)] = wl   
    FEATURES["zc_c"+str(j+1)] = zc 
    FEATURES["ssc_c"+str(j+1)] = ssc 
    # FEATURES["mav_c"+str(j+1)] = mav
    # FEATURES["ls_c"+str(j+1)] = ls
    # FEATURES["mfl_c"+str(j+1)] = mfl
    # FEATURES["msr_c"+str(j+1)] = msr
    # FEATURES["wamp_c"+str(j+1)] = wamp
    # FEATURES["iav_c"+str(j+1)] = iav
    # FEATURES["dasdv_c"+str(j+1)] = dasdv
    # FEATURES["var_c"+str(j+1)] = _var
label = np.zeros(l)
for i in range(l):
    if((etiqueta[i*I:(i)*I+W]>0).sum()>I/2):
        label[i] = max(etiqueta[i*I:(i)*I+W])
    else:
        label[i] = 0
# for i in range(l):
#     if((etiqueta[(i)*I+W-I:(i)*I+W]>0).sum()>I/2):
#         label[i] = max(etiqueta[i*I:(i)*I+W])
#     else:
#         label[i] = 0
label = pd.Series(label, dtype="category")

#### Split de datos
# Descarto el 50% del gesto 0
a, b = train_test_split(np.where(label==0)[0], test_size=5/10)
FEATURES.drop(a, inplace=True)
label.drop(a, inplace=True)    
FEATURES.index = np.arange(len(FEATURES))
label.index = np.arange(len(label))

sub_FEATURES, sub_label = features.sub_cat(FEATURES, label, 0)
sub_FEATURES.index = np.arange(len(sub_FEATURES))
sub_label.index = np.arange(len(sub_label))
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(sub_FEATURES, sub_label, test_size=1/7)

for i in range(1,21):
    sub_FEATURES, sub_label = features.sub_cat(FEATURES, label, i)
    X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(sub_FEATURES, sub_label, test_size=1/7)
    X_train_1 = pd.concat([X_train_1, X_train_aux])
    X_test_1 = pd.concat([X_test_1, X_test_aux])
    y_train_1 = pd.concat([y_train_1, y_train_aux])
    y_test_1 = pd.concat([y_test_1, y_test_aux])
X_train_1.index = np.arange(len(X_train_1))
X_test_1.index = np.arange(len(X_test_1))
y_train_1.index = np.arange(len(y_train_1))
y_test_1.index = np.arange(len(y_test_1))



tic = time()
red_2 = MLPClassifier(
      hidden_layer_sizes=(80,250,80),
      max_iter=100,
      activation='relu',
      validation_fraction=0.2, 
      early_stopping=True)
cv_results = cross_validate(red_2, X_train_1, y_train_1, cv=5)
# [['mfl_c1', 'var_c1', 'mfl_c2', 'msr_c2', 'wamp_c2', 'mfl_c3','dasdv_c3', 'mfl_c4', 'dasdv_c4', 'var_c4', 'mav_c5', 'dasdv_c5','var_c5', 'mfl_c6', 'mav_c7', 'mfl_c7', 'mfl_c8', 'var_c8','mfl_c9', 'mfl_c10', 'mfl_c11', 'msr_c11', 'rms_c12', 'mfl_c12','mfl_c13', 'dasdv_c13', 'mfl_c14', 'mfl_c15', 'msr_c15', 'mfl_c16']]

toc = time()
sorted(cv_results.keys())
print('Tiempo transcurrido: %.2f segundos' % (toc-tic))
print("Scores:")
print("Mean accuracy: %.2f" % (cv_results['test_score'].mean()*100))
print("Std accuracy: %.2f" % (cv_results['test_score'].std()*100))
print("Max accuracy: %.2f" % (cv_results['test_score'].max()*100))
print("Max fit time: %.2f" % cv_results['fit_time'].max())
