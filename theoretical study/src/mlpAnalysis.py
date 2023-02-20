# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:27:37 2023
In this script the analysis of the MLP model is carried out considering different architectures.
@Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""

from joblib import load
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../lib')
import classes as cl

FEATURES = load("../data/FEATURES.joblib")
label = load("../data/label.joblib")

data = cl.data(features = FEATURES, labels = label, chNum = 16, featureNum = 15)
X_train, X_validate, X_test, y_train, y_validate, y_test, = data.getSplitData(dropZeroSize = 8/10)
y_train = pd.Series(y_train["mov"], dtype="category")
y_test = pd.Series(y_test["mov"], dtype = "category")
y_validate = pd.Series(y_validate["mov"], dtype = "category")
nChannels = data.chNum
nFeatures = data.featureNum
inputDim = nChannels*nFeatures 
clasNum = data.clasNum

# Architecture 80 - 250 - 80
red = cl.Clasificador(model= "mlp", arch = [80,250,80], catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train, X_validate, y_validate)
y_test_predict = red.predict(X_test)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test_1 = red.confMatrix(X_test, y_test)

# Architecture 250 - 500 - 250
red = cl.Clasificador(model= "mlp", arch = [250,500,250], catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train, X_validate, y_validate)
y_test_predict = red.predict(X_test)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test_2 = red.confMatrix(X_test, y_test)

# Architecture 250 - 500 - 500 - 250
red = cl.Clasificador(model= "mlp", arch = [250,500,500,250], catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train, X_validate, y_validate)
y_test_predict = red.predict(X_test)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test_3 = red.confMatrix(X_test, y_test)

# Architecture 500 - 1000 - 500
red = cl.Clasificador(model= "mlp", arch = [500,1000,500], catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train, X_validate, y_validate)
y_test_predict = red.predict(X_test)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test_4 = red.confMatrix(X_test, y_test)


figure = plt.figure()
sns.heatmap(con_mat_test_1, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo de sujetos que no participan del entrenamiento")
plt.ylabel("Predict class")
plt.xlabel("True class")

figure = plt.figure()
sns.heatmap(con_mat_test_2, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo de sujetos que no participan del entrenamiento")
plt.ylabel("Predict class")
plt.xlabel("True class")

figure = plt.figure()
sns.heatmap(con_mat_test_3, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo de sujetos que no participan del entrenamiento")
plt.ylabel("Predict class")
plt.xlabel("True class")

figure = plt.figure()
sns.heatmap(con_mat_test_4, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo de sujetos que no participan del entrenamiento")
plt.ylabel("Predict class")
plt.xlabel("True class")

plt.show()