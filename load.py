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
eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]


    
EMG[:, 3] = (EMG[:,3]-np.mean(EMG[:,3]))/max(min(EMG[:, 3]), max(EMG[:, 3]))
EMG[:, 2] = (EMG[:,2]-np.mean(EMG[:,2]))/max(min(EMG[:, 2]), max(EMG[:, 2]))

ymin_1, ymax_1 =  min(EMG[:, 3]), max(EMG[:, 3])
ymin_2, ymax_2 =  min(EMG[:, 2]), max(EMG[:, 2])
EMG_fil = mm(EMG[:, 1])
fig, ax = plt.subplots(2, sharex=True, sharey=True)

ax[0].plot(t, EMG[:, 3])
ax[0].vlines(t[eventos_1], ymin_1, ymax_1, color = 'black')
ax[1].plot(t, EMG[:, 2])
ax[1].vlines(t[eventos_1], ymin_2, ymax_2, color = 'black')
ax[0].grid()
ax[1].grid()

# for i in range(int(len(eventos)/2)-1):    
#     ax[0].hlines(y=ymax_2, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
#     ax[0].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]], t[eventos_1[i*2]]))
#     ax[1].hlines(y=ymax_2, xmin=t[eventos[i*2]], xmax=t[eventos[i*2+1]], color = "black")
#     ax[1].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos[i*2]], ymax_2*1.1))
    
for i in range(int(len(eventos)/2)-1):    
    ax[0].hlines(y=ymax_2, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
    ax[0].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]], ymax_1*1.1))
    ax[1].hlines(y=ymax_2, xmin=t[eventos_1[i*2]], xmax=t[eventos_1[i*2+1]], color = "black")
    ax[1].annotate(str(etiqueta[eventos[i*2]+1]), (t[eventos_1[i*2]], ymax_2*1.1))   
    ax[0].grid()
    ax[1].grid()

# ax[1].set_xlim([10,80])
ax[1].set_xlabel("Tiempo (s)")
ax[1].set_ylabel("sEMG (u.a.)")
ax[0].set_ylabel("sEMG (u.a.)")
fig.suptitle("Señal de sEMG Sujeto 1. Canal 1 y 2.")



EMG[:, 1] = (EMG[:,1]-np.mean(EMG[:,1]))/max(min(EMG[:, 1]), max(EMG[:, 1]))
EMG[:, 2] = (EMG[:,2]-np.mean(EMG[:,2]))/max(min(EMG[:, 2]), max(EMG[:, 2]))

EMG_fil = mm(EMG[:, 1])
fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(t, EMG[:, 1], label = "Canal 1")
ax[1].plot(t, EMG[:, 2], label = "Canal 2")

ax[1].set_xlim([2.5,40.5])
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

