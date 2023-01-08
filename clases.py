# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:14:08 2022

@author: Renato
"""
from sklearn.svm import SVC
import lightgbm as lgb
import tensorflow as tf
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def re_label(labels, old_labels, new_label):
    for i in range(len(old_labels)):
        index = np.where(labels["mov"] == old_labels[i])[0]
        labels["mov"].iloc[index] = new_label
    return labels

class Clasificador:
    def __init__(self, model = "mlp", arch = [250,500,250], catNum = 52, featureNum = 4*16):
        self.model = model
        self.catNum = catNum
        self.featureNum = featureNum
        
        if(model == "mlp"):
            layers = [tf.keras.layers.Dense(arch[0], activation='relu', input_shape=(featureNum,))]
            for i in range(1,len(arch)-1):
                layers.append(tf.keras.layers.Dense(arch[i], activation='relu'))
            layers.append(tf.keras.layers.Dense(catNum, activation='softmax'))
            self.clf = tf.keras.Sequential(layers)
            self.clf.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
        
        elif(model == "svm"):
            self.clf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo', max_iter=(1000))
        
        elif(model == "gbm"):
            self.clf = lgb.LGBMClassifier()
            
        elif(model == "lda"):
            oa = OAS(store_precision=False, assume_centered=False)
            self.clf = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa)
            
    def train(self, xTrain, yTrain, epochs = 50):
        if(self.model == "mlp"):
            self.clf.fit(xTrain, yTrain, epochs = epochs)
        else:
            self.clf.fit(xTrain, yTrain)
    
    def predict(self, x):
        if(self.model == "mlp"):
            return self.clf.predict(x).argmax(axis=-1)
        else:
            return self.clf.predict(x)
        
    def confMatrix(self, x, yRef):
        yPredict = self.predict(x)
        conMat = metrics.confusion_matrix(yRef, yPredict, normalize="true")
        return pd.DataFrame(conMat, index = np.arange(self.catNum), columns = np.arange(self.catNum))       
        
class data:
    def __init__(self, features, labels, chNum, featureNum):
        self.features = features
        self.labels = labels
        self.chNum = chNum
        self.featureNum = featureNum
        self.clasNum = 52
    
    def modLabelWidth(self, width):
        eventos = np.where(np.diff(self.labels.T)[0]!=0)[0]
        for i in range(int(len(eventos)/2)):
            self.labels[eventos[i*2]:eventos[i*2]+width] = 0
            self.labels[eventos[(i+1)*2]-width:eventos[(i+1)*2]] = 0
    
    def selSubCat(self, subGroup = 1):      
        if(subGroup == 1): # Los primeros 12 movimientos (Ejercicio a)
            index = np.where(self.labels["mov"]==0)[0]
            for i in range(1,12):
                index = np.concatenate([index, np.where(self.labels["mov"]==i)[0]])
            self.features = self.features.iloc[index, :]
            self.labels = self.labels.iloc[index,:]
            self.features.index = np.arange(len(index))
            self.labels.index = np.arange(len(index))
            self.clasNum = 12
        elif(subGroup == 2): # 22 Movimientos (movimientos para hacer con la mano robótica)
            subCat = [20]
            cat = 52
            newLabel = re_label(self.labels, subCat, cat)
            
            subCat= [21,22,25,26,27,28]
            cat = 20
            newLabel = re_label(newLabel, subCat, cat)
            
            subCat = [30, 31,32,34,36,37,39,40,41,42]
            cat = 21 #Debe ir despues de asignar la cat_1 etiqueta 21 es "ficticia" equivale a la 30
            newLabel = re_label(newLabel, subCat, cat)
            
            subCat = [33]
            cat = 19
            newLabel = re_label(newLabel, subCat, cat)
            index = np.where(newLabel["mov"]==0)[0]
            for i in range(1,22):
                index = np.concatenate([index, np.where(newLabel["mov"]==i)[0]])
            self.features = self.features.iloc[index, :]
            self.labels = newLabel.iloc[index,:]
            self.features.index = np.arange(len(index))
            self.labels.index = np.arange(len(index))
            self.clasNum = 22
            
    def selSubCh(self, channels):
        chSelected = []
        for ch in self.features.columns:
            for i in channels:
                if(ch.split("_c")[1]==str(i)):
                    chSelected.append(ch)
        self.features = self.features[chSelected]
        self.chNum = len(channels)
        
    def selSubFeat(self, subFeat):
        featSelected = []
        for ch in self.features.columns:
            for i in subFeat:
                if(ch.split("_c")[0]==i):
                    featSelected.append(ch)
        self.features = self.features[featSelected]
        self.featureNum = len(subFeat)
    
    def getData(self):
        return self.features, self.labels
    
    def getSplitData(self, dropZeroSize = 1/2, testSize = 1/7):        
        a, b = train_test_split(np.where(self.labels["mov"]==0)[0], test_size=5/10)
        self.features.drop(a, inplace=True)
        self.labels.drop(a, inplace=True)    
        index = np.arange(self.labels.shape[0])
        self.features.index = index
        self.labels.index = index
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size = testSize)
        return X_train, X_test, y_train, y_test
        
   
        