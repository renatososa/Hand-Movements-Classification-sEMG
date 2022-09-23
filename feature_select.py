# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 19:39:47 2022

@author: Renato
"""


import pickle
import numpy as np
import extraccion_caract as features
import pandas as pd
import matplotlib.pyplot as plt

# Para crear redes neuronales:
from sklearn.neural_network import MLPClassifier

# Para dividir datos en entrenamiento/test:
from sklearn.model_selection import train_test_split


    
    
    
file = open('MyoArmband/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
fs = 200
w_ms = 1000 #Largo de la ventana
s_ms = 100 #Incremento
W = int(200*w_ms/1000)
I = int(200*s_ms/1000)
data = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']

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
label = pd.Series(label, dtype="category")



# Descarto el 50% del gesto 0
a, b = train_test_split(np.where(label==0)[0], test_size=5/10)
FEATURES.drop(a, inplace=True)
label.drop(a, inplace=True)    
FEATURES.index = np.arange(len(FEATURES))
label.index = np.arange(len(label))

sub_FEATURES, sub_label = features.sub_cat(FEATURES, label, 0)
sub_FEATURES.index = np.arange(len(sub_FEATURES))
sub_label.index = np.arange(len(sub_label))
X_train, X_test, y_train, y_test = train_test_split(sub_FEATURES, sub_label, test_size=1/7)

for i in range(1,21):
    sub_FEATURES, sub_label = features.sub_cat(FEATURES, label, i)
    X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(sub_FEATURES, sub_label, test_size=1/7)
    X_train = pd.concat([X_train, X_train_aux])
    X_test = pd.concat([X_test, X_test_aux])
    y_train = pd.concat([y_train, y_train_aux])
    y_test = pd.concat([y_test, y_test_aux])
X_train.index = np.arange(len(X_train))
X_test.index = np.arange(len(X_test))
y_train.index = np.arange(len(y_train))
y_test.index = np.arange(len(y_test))


red = MLPClassifier(
      hidden_layer_sizes=(80,250,80),
      max_iter=100,
      activation='relu',
      validation_fraction=0.2, 
      early_stopping=True)

select_features, scores, times, info, all_scores = features.SFC(red, X_train, y_train, splits = 5, n = 16, N_features = 3, by='channel')


# fig, ax = plt.subplots()
# bar_plot = plt.bar(np.arange(16.0), scores)
# for idx,rect in enumerate(bar_plot):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
#                 select_features[idx*3],
#                 ha='center', va='bottom', rotation=90)
#         ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
#                 round(scores[idx]),
#                 ha='center', va='bottom', rotation=0)
# ax.set_title("Accuracy al agregar canales")
fig, ax = plt.subplots(1)
bp = ax.boxplot(np.matrix(all_scores).T, whis = 2.7, showfliers = False)
plt.grid()