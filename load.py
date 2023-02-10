# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:01:06 2022

@author: Renato
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def mm(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

file = open('datos/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
correc = 200
EMG = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']
fs = MyoArm_data['S1']['info']['fs']
t = np.arange(len(EMG))/200
eventos_1 = np.where(np.diff(etiqueta.T)[0]!=0)[0]
eventos = np.zeros(len(eventos_1))
for i in range(int(len(eventos_1)/2)):
    etiqueta[eventos_1[i*2]:eventos_1[i*2]+correc] = 0
    etiqueta[eventos_1[i*2+1]-correc:eventos_1[i*2+1]+1] = 0
eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]

# 65160 66360 110940 112140


C1 = 5 -1 
C2 = 1 -1 
EMG[:, C1] = (EMG[:,C1]-np.mean(EMG[:,C1]))/max(min(EMG[:, C1]), max(EMG[:, C1]))
EMG[:, C2] = (EMG[:,C2]-np.mean(EMG[:,C2]))/max(min(EMG[:, C2]), max(EMG[:, C2]))

EMG_Export = np.concatenate([EMG[65160:66360, C1],EMG[114400:116600, C1]])
ymin_1, ymax_1 =  min(EMG[:, C1]), max(EMG[:, C1])
ymin_2, ymax_2 =  min(EMG[:, C2]), max(EMG[:, C2])


## Comparación de la corrección de las etiquetas
fig, ax = plt.subplots(2, sharex=True, sharey=True)

ax[0].plot(t, EMG[:, C1])
ax[0].vlines(t[eventos_1], ymin_1, ymax_1, color = 'black')
ax[1].plot(t, EMG[:, C1])
ax[1].vlines(t[eventos], ymin_1, ymax_1, color = 'black')
ax[0].grid()
ax[1].grid()

for i in range(int(len(eventos)/2)):    
    ax[0].hlines(y=ymax_1, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
    ax[0].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]]-1, ymax_1*1.1))
    ax[1].hlines(y=ymax_1, xmin=t[eventos[i*2]], xmax=t[eventos[i*2+1]], color = "black")
    ax[1].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos[i*2]]-1.5, ymax_1*1.1))
    ax[0].grid()
    ax[1].grid()


## Eventos con sus etiquetas canales 4 y 3
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
    


# ax[1].set_xlim([543,655])
ax[1].set_ylim([-1,2])
ax[1].set_xlabel("Tiempo (s)")
ax[1].set_ylabel("sEMG (u.a.)")
ax[0].set_ylabel("sEMG (u.a.)")
fig.suptitle("Señal de sEMG. Sensor 11 y 15. Movimientos 11 y 12.")
ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()


EMG[:, 1] = (EMG[:,1]-np.mean(EMG[:,1]))/max(min(EMG[:, 1]), max(EMG[:, 1]))
EMG[:, 2] = (EMG[:,2]-np.mean(EMG[:,2]))/max(min(EMG[:, 2]), max(EMG[:, 2]))


fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(t, EMG[:, 1], label = "Canal 1")
ax[1].plot(t, EMG[:, 2], label = "Canal 2")

# ax[1].set_xlim([2.5,40.5])
ax[1].set_xlabel("Tiempo (s)")
ax[1].set_ylabel("sEMG (u.a.)")
ax[0].set_ylabel("sEMG (u.a.)")
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
fig.suptitle("Señal de sEMG del Sujeto 1. Canal 1 y 2.")

plt.figure()
plt.plot(t, EMG[:, 1], label = "Canal 1")
plt.ylabel("sEMG (u.a.)")
plt.xlabel("Tiempo (s)")
plt.grid()
plt.legend()
plt.title("Señal de sEMG para el Sujeto 1. Canal 1. 3 repeticiones del movimiento 1.")

