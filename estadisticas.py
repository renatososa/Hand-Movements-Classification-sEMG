# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:46:46 2022

@author: Renato
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

file = open('datos/MyoArmband_data.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()


data = [MyoArm_data['S1']['emg'][:,0], MyoArm_data['S2']['emg'][:,0],
        MyoArm_data['S3']['emg'][:,0], MyoArm_data['S4']['emg'][:,0],
        MyoArm_data['S5']['emg'][:,0], MyoArm_data['S6']['emg'][:,0],
        MyoArm_data['S7']['emg'][:,0], MyoArm_data['S8']['emg'][:,0],
        MyoArm_data['S9']['emg'][:,0], MyoArm_data['S10']['emg'][:,0]]

data2 = []
mov = np.array([0])
for i in range(53):
    for x in range(1,11):
        ind = np.where(MyoArm_data['S'+str(x)]['label']==i)[0]
        mov = np.concatenate([mov,MyoArm_data['S'+str(x)]['emg'][:,0][ind]])
    data2.append(mov)    
    
fig, ax = plt.subplots(1)  
# Creating plot
bp = ax.boxplot(data, patch_artist = True, notch ='True', whis = 2.7, showfliers = False)
# x-axis labels
ax.set_xticklabels(['S1','S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'])
ax.set_xlabel("Sujetos")
ax.set_ylabel("EMG (mV)")
# Adding title
plt.title("EMG vs Sujeto (Electrodo 1)")

fig, ax = plt.subplots(1)  
# Creating plot
bp = ax.boxplot(data2, patch_artist = True, notch ='True', whis = 2.7, showfliers = False)
# x-axis labels
ax.set_xticklabels([str(x) for x in range(53)])

# Adding title
plt.title("EMG vs Movimiento (Electrodo 1)")