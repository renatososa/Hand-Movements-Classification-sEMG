# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:50:50 2022

@author: Renato
"""

import numpy as np
import extraccion_caract as features
import pandas as pd
from joblib import dump, load
# Para crear redes neuronales:
import tensorflow as tf

# Para dividir datos en entrenamiento/test:
from sklearn.model_selection import train_test_split


FEATURES = load("FEATURES.joblib")
label = load("label.joblib")

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

num_classes = 22
input_cat = 3*16
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

red = tf.keras.Sequential([tf.keras.layers.Dense(80, activation='relu', input_shape=(input_cat,)),
                           tf.keras.layers.Dense(250, activation='relu'),
                           tf.keras.layers.Dense(80, activation='relu'),
                          tf.keras.layers.Dense(num_classes, activation='softmax')])

red.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

red.fit(X_train, y_train, epochs=200)

print('Accuracy: %.2f' % red.evaluate(X_test, y_test)[1])

dump(X_train, "FEATURES.joblib")
dump(y_train, "label.joblib")
red.save("./")

