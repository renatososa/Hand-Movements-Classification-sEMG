# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:41:46 2022

@author: Renato
"""
import matplotlib.pyplot as plt
import numpy as np
import extraccion_caract as features
import seaborn as sns
from joblib import load
import clases as cl
import pandas as pd
from joblib import dump
import time

FEATURES = load("Datos/FEATURES.joblib")
label = load("Datos/label.joblib")

data = cl.data(features = FEATURES, labels = label, chNum = 16, featureNum = 15)
# data.selSubCat(subGroup=1)
# data.selSubCh([6,9,11,10,7,16,13,15]) # Los mejores canales para el mlp
# Split de los datos
X_train, X_test, y_train, y_test, = data.getSplitData(dropZeroSize = 8/10)
y_train = pd.Series(y_train["mov"], dtype="category")
y_test = pd.Series(y_test["mov"], dtype = "category")
nChannels = data.chNum
nFeatures = data.featureNum
inputDim = nChannels*nFeatures 
clasNum = data.clasNum
modelo = "mlp"




red = cl.Clasificador(model= modelo, catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train)



# channels, accuracyList = features.secuentialChannelSelection(X_train, y_train, X_test, y_test, nFeatures, nChannels, clasNum , model = modelo)
# feature, accuracyList = features.secuentialFeatureSelection(X_train, y_train, X_test, y_test, nFeatures, nFeatures, nChannels, clasNum , model = modelo)
# feature, accuracyList = features.accuracyByFeature(X_train, y_train, X_test, y_test, nFeatures, nChannels, clasNum , model = modelo)
# channels, accuracyList = features.accuracyByChannel(X_train, y_train, X_test, y_test, nFeatures,  clasNum = clasNum , model = modelo)

# dump([channels, accuracyList], "Datos/accuracyByChannelMLP.joblib")


# """Evaluamos modelo en conjunto de datos de **entrenamiento**:"""
# # Al sumar booleanos los True se toman como 1 y False como 0
y_train_predict = red.predict(X_train)
porcentaje_aciertos_train = 100*np.mean(y_train_predict==y_train)
print('Porcentaje de aciertos en conjunto de ENTRENAMIENTO: %.2f' % porcentaje_aciertos_train)

# """Evaluamos modelo en conjunto de datos de **test**:"""
a = time.time()
y_test_predict = red.predict(X_test)
a = a-time.time()
print(a)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)

con_mat_train = red.confMatrix(X_train, y_train)
con_mat_test = red.confMatrix(X_test, y_test)

figure = plt.figure()
sns.heatmap(con_mat_train, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de entrenamiento")
plt.ylabel("Predict class")
plt.xlabel("True class")

figure = plt.figure()
sns.heatmap(con_mat_test, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo")
plt.ylabel("Predict class")
plt.xlabel("True class")

plt.show()