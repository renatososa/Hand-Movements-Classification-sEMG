# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:37:42 2023

@author: Renato
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split

labels = load("Datos/label.joblib")["mov"]
labels_1 = labels.copy()

a, b = train_test_split(np.where(labels==0)[0], test_size=1/10)
labels_1.drop(a, inplace=True)    
index = np.arange(labels_1.shape[0])
labels_1.index = index

y_train, y_test = train_test_split(labels_1, test_size=2/7)


# Histograma de clases
plt.figure()
plt.hist(labels, np.arange(53.0+1.0), alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.title("Histogramas de datos sin descartar etiquetas")

plt.figure()
plt.hist(labels_1, np.arange(53.0+1.0), alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.title("Histogramas de datos descartando algunos datos de la etiqueta 'Reposo' ")

plt.figure()
plt.hist(y_train, np.arange(53.0+1.0), alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.title("Histogramas de datos para el conjunto de entrenamiento")

plt.figure()
plt.hist(y_test, np.arange(53.0+1.0), alpha=0.5, histtype='bar', ec='black')
plt.grid()
plt.title("Histogramas de datos para el conjunto de testeo")