# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 16:17:28 2023

@author: Renato
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import extraccion_caract as features
from sklearn.model_selection import train_test_split
import statistics as stat
import seaborn as sns
from everywhereml.code_generators.tensorflow import tf_porter
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics

## Labels: 0 - Rest, 1 - felexión meñique, 2 - extensión meñique, 3 -
  
dataMenique = np.loadtxt('menique.csv', dtype=float, delimiter=',').T
dataRing = np.loadtxt('ring.csv', dtype=float, delimiter=',').T
dataMedio = np.loadtxt('medio.csv', dtype=float, delimiter=',').T
dataIndice = np.loadtxt('indice.csv', dtype=float, delimiter=',').T
dataPulgar = np.loadtxt('pulgar.csv', dtype=float, delimiter=',').T
dataSupinacion = np.loadtxt('supinacion.csv', dtype=float, delimiter=',').T
dataPronacion = np.loadtxt('pronacion.csv', dtype=float, delimiter=',').T
dataPuno = np.loadtxt('puno.csv', dtype=float, delimiter=',').T

fs = 300
FEATURES = pd.DataFrame()
labels = np.concatenate([dataMenique[0], dataRing[0]*2, dataMedio[0]*3, dataIndice[0]*4, dataPulgar[0]*5, dataSupinacion[0]*6, dataPronacion[0]*7, dataPuno[0]*8])
wl = []
for i in range(8):
    FEATURES["wl_c"+str(i+1)] = np.concatenate([dataMenique[4*i+1]/100, dataRing[4*i+1]/100, dataMedio[4*i+1]/100, dataIndice[4*i+1]/100, dataPulgar[4*i+1]/100, dataSupinacion[4*i+1]/100, dataPronacion[4*i+1]/100, dataPuno[4*i+1]/100])
    FEATURES["zc_c"+str(i+1)] = np.concatenate([dataMenique[4*i+2]/100, dataRing[4*i+2]/100, dataMedio[4*i+2]/100, dataIndice[4*i+2]/100, dataPulgar[4*i+2]/100, dataSupinacion[4*i+2]/100, dataPronacion[4*i+2]/100, dataPuno[4*i+2]/100])
    FEATURES["ssc_c"+str(i+1)] = np.concatenate([dataMenique[4*i+3]/100, dataRing[4*i+3]/100, dataMedio[4*i+3]/100, dataIndice[4*i+3]/100, dataPulgar[4*i+3]/100, dataSupinacion[4*i+3]/100, dataPronacion[4*i+3]/100, dataPuno[4*i+3]/100])
    FEATURES["rms_c"+str(i+1)] = np.concatenate([dataMenique[4*i+4]/100, dataRing[4*i+4]/100, dataMedio[4*i+4]/100, dataIndice[4*i+4]/100, dataPulgar[4*i+4]/100, dataSupinacion[4*i+4]/100, dataPronacion[4*i+4]/100, dataPuno[4*i+4]/100])  
    

        
X_train, X_test, y_train, y_test = train_test_split(FEATURES, labels, test_size = 1/7)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3)


layers = [tf.keras.layers.Dense(80, activation='relu', input_shape=(32,))]
layers.append(tf.keras.layers.Dense(250, activation='relu'))
layers.append(tf.keras.layers.Dense(80, activation='relu'))
layers.append(tf.keras.layers.Dense(9, activation='softmax'))
model = tf.keras.Sequential(layers)
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_validate, y_validate))

porter = tf_porter(model, X_train, y_train)
cpp_code = porter.to_cpp(instance_name='mlp', arena_size=4096)

print(cpp_code)


y_train_predict = model.predict(X_train).argmax(axis=-1)
porcentaje_aciertos_train = 100*np.mean(y_train_predict==y_train)
print('Porcentaje de aciertos en conjunto de ENTRENAMIENTO: %.2f' % porcentaje_aciertos_train)

# """Evaluamos modelo en conjunto de datos de **test**:"""
y_test_predict = model.predict(X_test).argmax(axis=-1)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)


con_mat_train = metrics.confusion_matrix(y_train, y_train_predict, normalize="true")
con_mat_test = metrics.confusion_matrix(y_test, y_test_predict, normalize="true")

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
