# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:57:37 2023

@author: Renato
"""

import matplotlib.pyplot as plt
import numpy as np

teorico = np.loadtxt("EMG.csv")
flex_men = np.loadtxt("flex_menique.csv", skiprows=1, delimiter = ',').T[1]
flex_pulgar = np.loadtxt("flex_pulgar.csv", skiprows=1, delimiter = ',').T[1]

flex_men = flex_men/max(flex_men)/2
flex_pulgar = flex_pulgar/max(flex_men)/4

flex_men = flex_men[3500:7000]
flex_pulgar = flex_pulgar[1000:4500]

fig, ax = plt.subplots(2,2)

ax[0][0].plot(np.arange(1400)/200, teorico[:1400]*1.5, color='black', label = "Flexión del Meñique (Mov 7)")
ax[0][1].plot(np.arange(3500)/800, flex_men, label = "Flexión del Meñique (Mov 7)")
ax[1][0].plot(np.arange(1400)/200, teorico[1200:2600], color = 'black', label = "Flexión del pulgar (Mov 11)")
ax[1][1].plot(np.arange(3500)/800, flex_pulgar*1.5, label = "Flexión del Pulgar (Mov 11)")


ax[0][0].grid()
ax[0][1].grid()
ax[1][0].grid()
ax[1][1].grid()
ax[0][0].set_ylim(-1,1)
ax[0][1].set_ylim(-1,1)
ax[1][0].set_ylim(-1,1)
ax[1][1].set_ylim(-1,1)
ax[1][0].set_xlabel("Tiempo (s)")
ax[1][1].set_xlabel("Tiempo (s)")
ax[0][0].set_ylabel("Amplitud (u.a)")
ax[1][0].set_ylabel("Amplitud (u.a)")

fig.suptitle("Comparación de flexión de meñique y pulgar: base de datos vs sensores implementados")
ax[0][0].set_title("Base de datos")
ax[0][1].set_title("Sensor implementado")
ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()
