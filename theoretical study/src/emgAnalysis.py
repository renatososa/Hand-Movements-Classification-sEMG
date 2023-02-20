# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:01:06 2022
In this script a first approach to the signals and an analysis of the labels is made.
Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

file = open('../data/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
correc = 200
EMG = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']
fs = MyoArm_data['S1']['info']['fs']
t = np.arange(len(EMG))/200

# Corrección de las etiquetas
eventos_1 = np.where(np.diff(etiqueta.T)[0]!=0)[0]
eventos = np.zeros(len(eventos_1))
for i in range(int(len(eventos_1)/2)):
    etiqueta[eventos_1[i*2]:eventos_1[i*2]+correc] = 0
    etiqueta[eventos_1[i*2+1]-correc:eventos_1[i*2+1]+1] = 0
eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]

# Ploteo de las señales de sEMG
C1 = 1 - 1 # Sensor
C2 = 5 - 1 # Sensor
EMG[:, C1] = EMG[:,C1]/np.std(EMG[:, C1]) - np.mean(EMG[:, C1])
EMG[:, C2] = EMG[:,C2]/np.std(EMG[:, C2]) - np.mean(EMG[:, C2])

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(t, EMG[:, C1], label = "Sensor 1")
ax[1].plot(t, EMG[:, C2], label = "Sensor 5")
ax[1].set_xlabel("Tiempo (s)")
ax[1].set_ylabel("sEMG (u.a.)")
ax[0].set_ylabel("sEMG (u.a.)")
ax[0].grid(), ax[1].grid(), ax[0].legend(), ax[1].legend()
fig.suptitle("Señal de sEMG del Sujeto 1. Sensor 1 y 5.")

## Comparación de la corrección de las etiquetas
C1 = 5 -1 # Sensor
EMG[:, C1] = EMG[:,C1]/np.std(EMG[:, C1]) - np.mean(EMG[:, C1])
ymin_1, ymax_1 =  min(EMG[:, C1]), max(EMG[:, C1])

fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(t, EMG[:, C1], label = "Etiquetas corregidas")
ax[0].vlines(t[eventos_1], ymin_1, ymax_1, color = 'black')
ax[1].plot(t, EMG[:, C1], label = "Etiquetas sin corregir")
ax[1].vlines(t[eventos], ymin_1, ymax_1, color = 'black')

for i in range(int(len(eventos)/2)):    
    ax[0].hlines(y=ymax_1, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
    ax[0].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]]-1, ymax_1*1.1))
    ax[1].hlines(y=ymax_1, xmin=t[eventos[i*2]], xmax=t[eventos[i*2+1]], color = "black")
    ax[1].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos[i*2]]-1.5, ymax_1*1.1))

ax[0].grid(), ax[1].grid(), ax[0].legend(), ax[1].legend()  
ax[1].set_ylabel("sEMG (u.a.)"), ax[0].set_ylabel("sEMG (u.a.)")
ax[1].set_xlabel("Tiempo (s)")
ax[1].set_xlim([358,420])
fig.suptitle("Comparación de la corrección de las etiquetas. Sensor 5")

## Eventos con sus etiquetas sensores 11 y 15
C1 = 11 - 1 # Sensor
C2 = 15 - 1 # Sensor
EMG[:, C1] = EMG[:,C1]/np.std(EMG[:, C1]) - np.mean(EMG[:, C1])
EMG[:, C2] = EMG[:,C2]/np.std(EMG[:, C2]) - np.mean(EMG[:, C2])
ymin_1, ymax_1 =  min(EMG[:, C1]), max(EMG[:, C1])
ymin_2, ymax_2 =  min(EMG[:, C2]), max(EMG[:, C2])

fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(t, EMG[:, C1], label = "Sensor 11")
ax[0].vlines(t[eventos_1], ymin_1, ymax_1, color = 'black')
ax[1].plot(t, EMG[:, C2], label = "Sensor 15")
ax[1].vlines(t[eventos_1], ymin_2, ymax_2, color = 'black')

for i in range(int(len(eventos)/2)-1):    
    ax[0].hlines(y=ymax_1, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
    ax[0].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]], ymax_1*1.1))
    ax[1].hlines(y=ymax_2, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
    ax[1].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]], ymax_2*1.1))   

ax[1].set_xlim([543,655]), ax[1].set_ylim([-15,25])
ax[1].set_ylabel("sEMG (u.a.)"), ax[0].set_ylabel("sEMG (u.a.)")
ax[1].set_xlabel("Tiempo (s)")
ax[0].grid(), ax[1].grid(), ax[0].legend(), ax[1].legend()
fig.suptitle("Señal de sEMG. Sensor 11 y 15. Movimientos 11 y 12.")


plt.show()


