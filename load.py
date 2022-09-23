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

file = open('MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()

EMG = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']
fs = MyoArm_data['S1']['info']['fs']
t = np.arange(len(EMG))/200
eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]
for i in range(int(len(eventos)/2)):
    etiqueta[eventos[i*2]:eventos[i*2]+50] = 0
    etiqueta[eventos[(i+1)*2]-50:eventos[(i+1)*2]] = 0
eventos = np.where(np.diff(etiqueta.T)[0]!=0)[0]


    
EMG[:, 0] = (EMG[:,0]-np.mean(EMG[:,0]))/max(min(EMG[:, 0]), max(EMG[:, 0]))
EMG[:, 1] = (EMG[:,1]-np.mean(EMG[:,1]))/max(min(EMG[:, 1]), max(EMG[:, 1]))

ymin_1, ymax_1 =  min(EMG[:, 0]), max(EMG[:, 0])
ymin_2, ymax_2 =  min(EMG[:, 1]), max(EMG[:, 1])
EMG_fil = mm(EMG[:, 1])
fig, ax = plt.subplots(2, sharex=True, sharey=True)

ax[0].plot(t, EMG[:, 0])
ax[0].vlines(t[eventos], ymin_1, ymax_1, color = 'black')
ax[1].plot(t, EMG[:, 1])
ax[1].vlines(t[eventos], ymin_2, ymax_2, color = 'black')

for i in range(int(len(eventos)/2)-1):    
    ax[0].hlines(y=ymax_1, xmin=t[eventos[i*2]], xmax=t[eventos[i*2+1]], color = "black")
    ax[0].annotate(str(etiqueta[eventos[i*2+1]]), (t[eventos[i*2]], ymax_1*1.1))
    ax[1].hlines(y=ymax_2, xmin=t[eventos[i*2]], xmax=t[eventos[i*2+1]], color = "black")
    ax[1].annotate(str(etiqueta[eventos[i*2+1]]), (t[eventos[i*2]], ymax_2*1.1))

# ax[0].legend([str(x) for x in range(16)])

