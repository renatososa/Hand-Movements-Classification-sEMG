# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:41:46 2022

@author: Renato
"""
import matplotlib.pyplot as plt
import numpy as np
import extraccion_caract as features
import pandas as pd
import seaborn as sns
from joblib import load

def mediana(data):
    out = data.copy()
    for i in range(len(data)-3):
        out[i] = np.median(data[i:i+3])
    return out

import tensorflow as tf

# Para crear redes neuronales:
from sklearn.neural_network import MLPClassifier

# Para dividir datos en entrenamiento/test:
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.utils import shuffle

# Para medir tiempos
from time import time

FEATURES = load("FEATURES.joblib")
label = load("label.joblib")

sub_cat_1 = [2,4,6,8,9]
cat_1 = 10
sub_cat_2 = [12]
cat_2 = 2
sub_cat_3 = [23]
cat_3 = 4 #Debe ir despues de asignar la cat_1 etiqueta 21 es "ficticia" equivale a la 30
sub_cat_4 = [24]
cat_4 = 6

label = features.re_label(label, sub_cat_1, cat_1)
label = features.re_label(label, sub_cat_2, cat_2)
label = features.re_label(label, sub_cat_3, cat_3)
label = features.re_label(label, sub_cat_4, cat_4)

#### Split de datos
# Descarto el 50% del gesto 0
a, b = train_test_split(np.where(label==0)[0], test_size=5/10)
FEATURES.drop(a, inplace=True)
label.drop(a, inplace=True)    
FEATURES.index = np.arange(len(FEATURES))
label.index = np.arange(len(label))

sub_FEATURES, sub_label = features.sub_cat(FEATURES, label, 0)
sub_FEATURES.index = np.arange(len(sub_FEATURES))
sub_label.index = np.arange(len(sub_label))
X_train, X_test, y_train, y_test = train_test_split(sub_FEATURES, sub_label, test_size=1/7)

num_classes = 8
input_cat = 6*16
movs = [1,3,5,7,12,23,24]


for i in range(1,num_classes):
    sub_FEATURES, sub_label = features.sub_cat(FEATURES, label, i)
    X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(sub_FEATURES, sub_label, test_size=1/7)
    X_train = pd.concat([X_train, X_train_aux])
    X_test = pd.concat([X_test, X_test_aux])
    y_train = pd.concat([y_train, y_train_aux])
    y_test = pd.concat([y_test, y_test_aux])
X_train.index = np.arange(len(X_train))
X_test.index = np.arange(len(X_test))
y_train.index = np.arange(len(y_train))
y_test.index = np.arange(len(y_test))
# Histograma de clases



red = tf.keras.Sequential([tf.keras.layers.Dense(250, activation='relu', input_shape=(input_cat,)),
                            tf.keras.layers.Dense(500, activation='relu'),
                            tf.keras.layers.Dense(500, activation='relu'),
                            tf.keras.layers.Dense(250, activation='relu'),
                          tf.keras.layers.Dense(num_classes, activation='softmax')])

red.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

red.fit(X_train, y_train, epochs=50)

print('Accuracy: %.2f' % red.evaluate(X_test, y_test)[1])


# """Evaluamos modelo en conjunto de datos de **entrenamiento**:"""

# # Al sumar booleanos los True se toman como 1 y False como 0
y_train_predict = red.predict(X_train).argmax(axis=-1)
porcentaje_aciertos_train = 100*np.mean(y_train_predict==y_train)
print('Porcentaje de aciertos en conjunto de ENTRENAMIENTO: %.2f' % porcentaje_aciertos_train)

# """Evaluamos modelo en conjunto de datos de **test**:"""


y_test_predict = red.predict(X_test).argmax(axis=-1)
y_predict_fil = mediana(y_test_predict)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)

porcentaje_aciertos_test_fil = 100*np.mean(y_predict_fil==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test_fil)


con_mat_train = metrics.confusion_matrix(y_train, y_train_predict, normalize="true")
con_mat_test = metrics.confusion_matrix(y_test, y_test_predict, normalize="true")
con_mat_test_fil = metrics.confusion_matrix(y_test, y_predict_fil, normalize="true")
con_mat_train = pd.DataFrame(con_mat_train, index = np.arange(num_classes), columns = np.arange(num_classes))
con_mat_test = pd.DataFrame(con_mat_test, index = np.arange(num_classes), columns = np.arange(num_classes))
con_mat_test_fil = pd.DataFrame(con_mat_test_fil, index = np.arange(num_classes), columns = np.arange(num_classes))

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

figure = plt.figure()
sns.heatmap(con_mat_test_fil, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6})
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo filtrados")
plt.ylabel("Predict class")
plt.xlabel("True class")

