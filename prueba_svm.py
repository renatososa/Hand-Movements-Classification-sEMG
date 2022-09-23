# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:41:00 2022

@author: Renato
"""
import matplotlib.pyplot as plt
import numpy as np
import extraccion_caract as features
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.svm import SVC

def mediana(data):
    out = data.copy()
    for i in range(len(data)-3):
        out[i] = np.median(data[i:i+3])
    return out


# Para dividir datos en entrenamiento/test:
from sklearn.model_selection import train_test_split

from sklearn import metrics

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
input_cat = 11*16
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

rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo', max_iter=(1000))
rbf.fit(X_train, y_train)

y_test_predict = rbf.predict(X_test)
y_train_predict = rbf.predict(X_train)

porcentaje_aciertos_train = 100*np.mean(y_train_predict==y_train)
print('Porcentaje de aciertos en conjunto de ENTRENAMIENTO: %.2f' % porcentaje_aciertos_train)
y_predict_fil = mediana(y_test_predict)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)

porcentaje_aciertos_test_fil = 100*np.mean(y_predict_fil==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test_fil)


con_mat_train = metrics.confusion_matrix(y_train, y_train_predict, normalize="true")
con_mat_test = metrics.confusion_matrix(y_test, y_test_predict, normalize="true")
con_mat_test_fil = metrics.confusion_matrix(y_test, y_predict_fil, normalize="true")
con_mat_train = pd.DataFrame(con_mat_train, index = np.arange(22), columns = np.arange(22))
con_mat_test = pd.DataFrame(con_mat_test, index = np.arange(22), columns = np.arange(22))
con_mat_test_fil = pd.DataFrame(con_mat_test_fil, index = np.arange(22), columns = np.arange(22))

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