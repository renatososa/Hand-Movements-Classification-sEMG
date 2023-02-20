# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:26:49 2023

@Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

nMovs = 9
x_axis_labels = ["Reposo","Meñique","Anular","Medio","Índice","Pulgar","Pronación","Supinación","Cerrada"] 
data = np.loadtxt('../data/eval_1.csv', dtype=float, delimiter=',').T[:2]
for i in range(2, nMovs):
    data_aux = np.loadtxt("../data/eval_"+str(i)+".csv", dtype=float, delimiter=',').T[:2]
    data_aux[0] = data_aux[0]*i
    data = np.concatenate((data, data_aux), axis = 1)


con_mat_eval = metrics.confusion_matrix(data[0], data[1], normalize="true")

figure = plt.figure()
sns.heatmap(con_mat_eval, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6}, xticklabels=x_axis_labels, yticklabels=x_axis_labels)
plt.tight_layout()
plt.title("Matriz de confusión para clasificación en el ESP32")
plt.ylabel("Predict class")
plt.xlabel("True class")

dataMenique = np.loadtxt('../data/eval_1.csv', dtype=float, delimiter=',').T
plt.figure()
plt.title("Predicción de la flexión del dedo meñique en el ESP32")
plt.plot(np.arange(len(dataMenique[1]))/40, dataMenique[1], "-.o", label = "Predicción")
plt.plot(np.arange(len(dataMenique[1]))/40, dataMenique[0], "-.*", label = "Etiqueta", ms = 4)
plt.grid()
plt.legend()
plt.yticks(np.arange(len(x_axis_labels)), x_axis_labels)
plt.ylabel("Movimiento")
plt.xlabel("Tiempo (s)")

dataMedio = np.loadtxt('../data/eval_3.csv', dtype=float, delimiter=',').T
plt.figure()
plt.title("Predicción de la flexión del dedo medio en el ESP32")
plt.plot(np.arange(len(dataMedio[1]))/40, dataMedio[1], "-.o", label = "Predicción")
plt.plot(np.arange(len(dataMedio[1]))/40, dataMedio[0]*3, "-.*", label = "Etiqueta", ms = 4)
plt.grid()
plt.legend()
plt.yticks(np.arange(len(x_axis_labels)), x_axis_labels)
plt.ylabel("Movimiento")
plt.xlabel("Tiempo (s)")
