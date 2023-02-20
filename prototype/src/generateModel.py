# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 16:17:28 2023
Script to design, train and export ML model.
@Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from everywhereml.code_generators.tensorflow import tf_porter
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics


nMovs = 9
data = np.loadtxt("../data/1.csv", dtype=float, delimiter=',').T
for i in range(2,nMovs):
    data_aux = np.loadtxt("../data/" + str(i) + ".csv", dtype=float, delimiter=',').T
    data_aux[0] = data_aux[0]*i
    data = np.concatenate((data, data_aux), axis = 1)

labels = data[0]
FEATURES = pd.DataFrame()
for i in range(8):
    FEATURES["wl_c"+str(i+1)] = data[4*i+1]/100
    FEATURES["zc_c"+str(i+1)] = data[4*i+2]/100
    FEATURES["ssc_c"+str(i+1)] = data[4*i+3]/100
    FEATURES["rms_c"+str(i+1)] = data[4*i+4]/100

X_train, X_test, y_train, y_test = train_test_split(FEATURES, labels, test_size = 1/7)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3)

# Design models
layers = [tf.keras.layers.Dense(80, activation='relu', input_shape=(32,))]
layers.append(tf.keras.layers.Dense(250, activation='relu'))
layers.append(tf.keras.layers.Dense(80, activation='relu'))
layers.append(tf.keras.layers.Dense(9, activation='softmax'))
model = tf.keras.Sequential(layers)
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_validate, y_validate))

# Export model
porter = tf_porter(model, X_train, y_train)
cpp_code = porter.to_cpp(instance_name='mlp', arena_size=4096)

text_file = open("../ESP32_MLP/src/model.h", "w")
text_file.write(cpp_code)
text_file.close()

# Luego de generar el archivo model.h, modificar la línea 39: Eloquent::TinyML::TensorFlow::AllOpsTensorFlow<32, 1, arenaSize> tf;
# por: Eloquent::TinyML::TensorFlow::AllOpsTensorFlow<32, 9, arenaSize> tf;


y_test_predict = model.predict(X_test).argmax(axis=-1)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test = metrics.confusion_matrix(y_test, y_test_predict, normalize="true")

x_axis_labels = ["Reposo","Meñique","Anular","Medio","Índice","Pulgar","Pronación","Supinación","Cerrada"] 

figure = plt.figure()
sns.heatmap(con_mat_test, annot=True, cmap=plt.cm.Blues, annot_kws={"fontsize":6}, xticklabels=x_axis_labels, yticklabels=x_axis_labels)
plt.tight_layout()
plt.title("Matriz de confusión para los datos de testeo")
plt.ylabel("Predict class")
plt.xlabel("True class")
plt.show()


