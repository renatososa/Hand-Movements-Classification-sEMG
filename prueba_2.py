# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:41:46 2022

@author: Renato
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import extraccion_caract as features
import pandas as pd
import seaborn as sns

# Para crear redes neuronales:
from sklearn.neural_network import MLPClassifier

# Para dividir datos en entrenamiento/test:
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.model_selection import cross_validate

# Para medir tiempos
from time import time

file = open('MyoArmband/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
fs = 200
w_ms = 1000 #Largo de la ventana
s_ms = 25 #Incremento
W = int(200*w_ms/1000)
I = int(200*s_ms/1000)
data = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']

for j in range(16):
    data[:,j] = data[:,j]/max(np.max(data[:,j]), abs(np.min(data[:,j]))) 

for i in range(2, 6):
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
        rms[i] = features.rms(data[:,j][i*I:(i)*I+W])
        wl[i] = features.wl(data[:,j][i*I:(i)*I+W])
        zc[i] = features.zc(data[:,j][i*I:(i)*I+W], 0.01)
        ssc[i] = features.ssc(data[:,j][i*I:(i)*I+W], 0.01)
        mav[i] = features.mav(data[:,j][i*I:(i)*I+W])
        ls[i] = features.ls(data[:,j][i*I:(i)*I+W])
        mfl[i] = features.mfl(data[:,j][i*I:(i)*I+W])
        msr[i] = features.msr(data[:,j][i*I:(i)*I+W])
        wamp[i] = features.wamp(data[:,j][i*I:(i)*I+W])
        iav[i] = features.iav(data[:,j][i*I:(i)*I+W])
        dasdv[i] = features.dasdv(data[:,j][i*I:(i)*I+W])
        _var[i] = features._var(data[:,j][i*I:(i)*I+W])
    FEATURES["rms_c"+str(j+1)] = rms       
    FEATURES["wl_c"+str(j+1)] = wl   
    FEATURES["zc_c"+str(j+1)] = zc 
    FEATURES["ssc_c"+str(j+1)] = ssc 
    FEATURES["mav_c"+str(j+1)] = mav
    FEATURES["ls_c"+str(j+1)] = ls
    FEATURES["mfl_c"+str(j+1)] = mfl
    FEATURES["msr_c"+str(j+1)] = msr
    FEATURES["wamp_c"+str(j+1)] = wamp
    FEATURES["iav_c"+str(j+1)] = iav
    FEATURES["dasdv_c"+str(j+1)] = dasdv
    FEATURES["var_c"+str(j+1)] = _var
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
# Sacar muestras de clase 0 o duplicar las otras clases



# Hacer split por evento y hacer histograma. Dejar sujeto fuera del conjunto de train.
X_train, X_test, y_train, y_test = train_test_split(FEATURES, label, test_size=1/7)
# Histograma de clases

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist(y_train, np.arange(54.0), alpha=0.5, histtype='bar', ec='black')
axs[0].set_title("Histograma de clases con train_test_split")
axs[1].hist(y_train_1, np.arange(54.0), alpha=0.5, histtype='bar', ec='black')
axs[1].set_title("Histograma de clases separando por clases")


#################################################################################################
# Entrenamos red neuronal en Scikit-learn:
tic = time()
red_2 = MLPClassifier(
      hidden_layer_sizes=(250,500,250),
      max_iter=100,
      activation='relu',
      validation_fraction=0.2, 
      early_stopping=True)
red_2.fit(X_train_1, y_train_1)
toc = time()

print('Tiempo transcurrido: %.2f segundos' % (toc-tic))

# """Evaluamos modelo en conjunto de datos de **entrenamiento**:"""

# # Al sumar booleanos los True se toman como 1 y False como 0
y_train_predict = red_2.predict(X_train_1)
porcentaje_aciertos_train = 100*np.mean(y_train_predict==y_train_1)
print('Porcentaje de aciertos en conjunto de ENTRENAMIENTO: %.2f' % porcentaje_aciertos_train)

# """Evaluamos modelo en conjunto de datos de **test**:"""


y_test_predict = red_2.predict(X_test_1)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test_1)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)

# """Evolución de la función de costo durante el entrenamiento:"""

plt.figure(figsize=(10,5))
plt.plot(red_2.loss_curve_);

con_mat_train = metrics.confusion_matrix(y_train_1, y_train_predict, normalize="true")
con_mat_test = metrics.confusion_matrix(y_test_1, y_test_predict, normalize="true")
con_mat_train = pd.DataFrame(con_mat_train, index = np.arange(21), columns = np.arange(21))
con_mat_test = pd.DataFrame(con_mat_test, index = np.arange(21), columns = np.arange(21))

figure = plt.figure()
sns.heatmap(con_mat_train, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.title("Matriz de confusión para los datos de entrenamiento")
plt.ylabel("Predict class")
plt.xlabel("True class")

figure = plt.figure()
sns.heatmap(con_mat_test, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo")
plt.ylabel("Predict class")
plt.xlabel("True class")


# S_test = MyoArm_data['S1']['emg']
# etiqueta = MyoArm_data['S1']['label']
# # for i in range(6, 11):    
# #     etiqueta = np.concatenate((etiqueta, MyoArm_data['S'+str(i)]['label']))
# #     S_test = np.concatenate((S_test, MyoArm_data['S'+str(i)]['emg']))
# l = int((len(S_test)-W+S)/S)

# # Corrección de etiquetas
# eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]
# for i in range(int(len(eventos)/2)):
#     etiqueta[eventos[i*2]:eventos[i*2]+50] = 0
#     etiqueta[eventos[(i+1)*2]-50:eventos[(i+1)*2]] = 0

# # Normalizacion
# for j in range(16):
#     S_test[:,j] = S_test[:,j]/max(np.max(S_test[:,j]), abs(np.min(S_test[:,j]))) 

# rms = np.zeros(l)
# mav = np.zeros(l)
# mavs = np.zeros(l)
# zc = np.zeros(l)
# ssc = np.zeros(l)

# # Ver algoritmos de selección de características
# FEATURES_2 = pd.DataFrame()
# for j in range(16):
#     for i in range(l):
#         rms[i] = features.rms(S_test[:,j][i*S:(i)*S+W])
#         mav[i] = features.mav(S_test[:,j][i*S:(i)*S+W])
#         mavs[i] = features.mavs(S_test[:,j][i*S:(i)*S+W])
#         zc[i] = features.zc(S_test[:,j][i*S:(i)*S+W], 0.01)
#         ssc[i] = features.ssc(S_test[:,j][i*S:(i)*S+W], 0.01)
#     FEATURES_2["rms_c"+str(j+1)] = rms
#     FEATURES_2["mav_c"+str(j+1)] = mav
#     FEATURES_2["mavs_c"+str(j+1)] = mavs
#     FEATURES_2["zc_c"+str(j+1)] = zc
#     FEATURES_2["ssc_c"+str(j+1)] = ssc
# label = np.zeros(l)
# for i in range(l):
#     if((etiqueta[i*S:(i)*S+W]>0).sum()>S/2):
#         label[i] = max(etiqueta[i*S:(i)*S+W])
#     else:
#         label[i] = 0
# label = pd.Series(label, dtype="category")



# y_test_predict = red_2.predict(FEATURES_2)
# porcentaje_aciertos_test = 100*np.mean(y_test_predict==label)
# print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
# con_mat_test = metrics.confusion_matrix(label, y_test_predict, normalize="true")
# con_mat_test = pd.DataFrame(con_mat_test, index = np.arange(53), columns = np.arange(53))

# figure = plt.figure()
# sns.heatmap(con_mat_test, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
# plt.tight_layout()
# plt.title("Matriz de confusión para los sujetos de testeo 6, 7, 8, 9 y 10")
# plt.ylabel("Predict class")
# plt.xlabel("True class")