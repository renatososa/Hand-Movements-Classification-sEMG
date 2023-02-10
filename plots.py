# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 08:47:36 2022

@author: Renato
"""

import matplotlib.pyplot as plt
from joblib import load
import numpy as np


# =============================================================================
# Accuracy por feature para las 16 canales
accuracyByFeatureLDA = load("datos/accuracyByFeatureLDA.joblib")
accuracyByFeatureSVM = load("datos/accuracyByFeatureSVM.joblib")
accuracyByFeatureMLP = load("datos/accuracyByFeatureMLP.joblib")
accuracyByFeatureGBM = load("datos/accuracyByFeatureGBM.joblib")

X = accuracyByFeatureLDA[0]
X_axis = np.arange(15)*2
plt.figure()
# plt.bar(X_axis - 0.6, np.array(accuracyByFeatureLDA[1]), 0.4, label = 'LDA')
plt.bar(X_axis - 0.2, np.array(accuracyByFeatureSVM[1]), 0.4, label = 'SVM')
plt.bar(X_axis + 0.2, np.array(accuracyByFeatureGBM[1]), 0.4, label = 'GBM')
plt.bar(X_axis + 0.6, np.array(accuracyByFeatureMLP[1]), 0.4, label = 'MLP')

plt.ylim(60,100)
plt.xticks(X_axis, X)
plt.xlabel("Características")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy para cada característica (16 sensores)")
plt.grid()
plt.legend()
plt.show()

# =============================================================================
# Accuracy por canal para las 4 features
accuracyByFeatureSVM = load("datos/accuracyByChannelSVM.joblib")
accuracyByFeatureMLP = load("datos/accuracyByChannelMLP.joblib")
accuracyByFeatureGBM = load("datos/accuracyByChannelGBM.joblib")
X = np.arange(1, 17)
X_axis = np.arange(16)*2
plt.figure()
# plt.bar(X_axis - 0.6, np.array(accuracyByFeatureLDA[1]), 0.4, label = 'LDA')
plt.bar(X_axis - 0.2, np.array(accuracyByFeatureSVM[1][:-1])*100, 0.4, label = 'SVM')
plt.bar(X_axis + 0.2, np.array(accuracyByFeatureGBM[1][:-1])*100, 0.4, label = 'GBM')
plt.bar(X_axis + 0.6, np.array(accuracyByFeatureMLP[1][:-1])*100, 0.4, label = 'MLP')

plt.ylim(60,75)
plt.xticks(X_axis, X)
plt.xlabel("N° Sensor")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy para cada sensor (rms, wl, zc, ssc)")
plt.grid()
plt.legend()
plt.show()
# =============================================================================
# =============================================================================
# Accuracy en función de la cantidad de canales (rms, wl, zc, ssc. Selección secuencial)

channelAnalisisLDA = load("datos/accuracyPerChannelLDA.joblib")
channelAnalisisSVM = load("datos/accuracyPerChannelSVM.joblib")
channelAnalisisMLP = load("datos/accuracyPerChannelMLP.joblib")
channelAnalisisGBM = load("datos/accuracyPerChannelGBM.joblib")

plt.figure()  
# plt.plot(np.array(channelAnalisisLDA[1])*100,"-o", label = 'LDA')
plt.plot(np.array(channelAnalisisSVM[1])*100,"-o", label = 'SVM')
plt.plot(np.array(channelAnalisisGBM[1])*100,"-o", label = 'GBM')
plt.plot(np.array(channelAnalisisMLP[1])*100,"-o", label = 'MLP')

plt.grid()
plt.xlabel("Número de canales")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy en función de la cantidad de sensores (rms, wl, zc, ssc)")
plt.legend()
plt.ylim(60,100)
plt.show()
# =============================================================================
# =============================================================================
# Ubicación de los 8 mejores sensores para MLP y GBM 
canalesGBM = np.zeros(8)
canalesMLP = np.zeros(8)
for i in range(8): 
    canalesGBM[i] = int(channelAnalisisGBM[0][i*4].split('_c')[1])
    canalesMLP[i] = int(channelAnalisisMLP[0][i*4].split('_c')[1])

canalesGBM_1 = canalesGBM[np.where(canalesGBM<9)[0]]
canalesGBM_2 = canalesGBM[np.where(canalesGBM>8)[0]]-8
canalesMLP_1 = canalesMLP[np.where(canalesMLP<9)[0]]
canalesMLP_2 = canalesMLP[np.where(canalesMLP>8)[0]]-8
canalesGBM_1 = canalesGBM_1*2*np.pi/8 -np.pi/2
canalesGBM_2 = canalesGBM_2*2*np.pi/8 -np.pi/2 + np.pi*22/180
canalesMLP_1 = canalesMLP_1*2*np.pi/8 -np.pi/2
canalesMLP_2 = canalesMLP_2*2*np.pi/8 -np.pi/2 + np.pi*22/180

circulo = np.arange(1,1000)*np.pi*2/1000
xCirculoGBM = np.cos(canalesGBM)
yCirculoGBM = np.sin(canalesGBM)

fig, ax = plt.subplots(1,2, sharex = True, sharey=True) 
ax[0].plot(np.cos(circulo), np.sin(circulo), 'grey')
ax[0].plot(np.cos(canalesGBM_1), np.sin(canalesGBM_1), 'ok', label = "GBM")
ax[0].plot(np.cos(canalesMLP_1), np.sin(canalesMLP_1), '*r', label = "MLP")
ax[1].plot(np.cos(circulo), np.sin(circulo), 'grey')
ax[1].plot(np.cos(canalesGBM_2), np.sin(canalesGBM_2), 'ok', label = "GBM")
ax[1].plot(np.cos(canalesMLP_2), np.sin(canalesMLP_2), '*r', label = "MLP")
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[0].set_title("Banda 1")
ax[1].set_title("Banda 2")
ax[0].text(-0.25, -1.5, r'Codo', fontsize=15)
ax[1].text(-0.25, -1.5, r'Codo', fontsize=15)
fig.suptitle("Distribución de 8 sensores para la mejor accuracy (rms, wl, zc, ssc)")
plt.show()
# =============================================================================
# =============================================================================
# Accuracy en función de la cantidad de caracterpiticas (16 sensores, selección secuencial)
featureAnalisisLDA = load("datos/accuracyPerFeatureLDA.joblib")
featureAnalisisSVM = load("datos/accuracyPerFeatureSVM.joblib")
featureAnalisisMLP = load("datos/accuracyPerFeatureMLP.joblib")
featureAnalisisGBM = load("datos/accuracyPerFeatureGBM.joblib")
featureAnalisisMLP8 = load("datos/accuracyPerFeatureMLP8.joblib")


plt.figure()
# plt.plot(np.array(featureAnalisisLDA[1])*100,"-o", label = 'LDA')
plt.plot(np.arange(1,16), np.array(featureAnalisisSVM[1])*100,"-o", label = 'SVM')
plt.plot(np.arange(1,16), np.array(featureAnalisisGBM[1])*100,"-o", label = 'GBM')
plt.plot(np.arange(1,16), np.array(featureAnalisisMLP[1])*100,"-o", label = 'MLP')

plt.grid()
plt.xlabel("Número de caracteristicas")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy en función de la cantidad de caracteristicas (16 sensores)")
plt.legend()
plt.ylim(80,100)
plt.show()
# =============================================================================

plt.figure()
plt.plot(np.array(featureAnalisisMLP8[1])*100,"-or", label = 'MLP')

plt.grid()
plt.xlabel("Número de caracteristicas")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy en función de la cantidad de caracteristicas (8 mejores sensores)")
plt.legend()
plt.ylim(80,100)
plt.show()

# =============================================================================

fig, ax = plt.subplots(1,2, sharex = True, sharey=True) 

circulo = np.arange(1,1000)*np.pi*2/1000
Myo_banda1 = np.arange(8)*2*np.pi/8 -np.pi/2
Myo_banda2 = np.arange(8)*2*np.pi/8 -np.pi/2 + np.pi*22/180
ax[0].plot(np.cos(circulo), np.sin(circulo), 'grey', label = "Perímetro del antebrazo")
ax[0].plot(np.cos(circulo)*0.9, np.sin(circulo)*0.9, 'grey')
ax[0].plot(np.cos(Myo_banda1), np.sin(Myo_banda1), 'ok', label = "Sensores Banda 1")
ax[0].plot(np.cos(Myo_banda2)*0.9, np.sin(Myo_banda2)*0.9, 'or', label = "Sensores Banda 2")
ax[1].plot(np.cos(circulo), np.sin(circulo), 'grey', label = "Perímetro del antebrazo")
ax[1].plot(np.cos(circulo)*0.9, np.sin(circulo)*0.9, 'grey')
ax[1].plot(np.cos(Myo_banda1[[1,3,5,7]]), np.sin(Myo_banda1[[1,3,5,7]]), 'ok', label = "Sensores Banda 1")
ax[1].plot(np.cos(Myo_banda2[[0,2,4,6]])*0.9, np.sin(Myo_banda2[[0,2,4,6]])*0.9, 'or', label = "Sensores Banda 2")
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_title("Myo Armband")
ax[1].set_title("Banda diseñada")
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
