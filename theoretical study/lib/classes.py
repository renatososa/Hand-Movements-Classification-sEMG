# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:14:08 2022
Classes difinition
@Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""
from sklearn.svm import SVC
import lightgbm as lgb
import tensorflow as tf
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Clasificador:
    """
    class to manage the clasification: design, train and predict.
    """
    def __init__(self, model = "mlp", arch = [250,500,250], catNum = 53, featureNum = 4*16):
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
                
    def train(self, xTrain, yTrain, X_validate, y_validate, epochs = 50):
        if(self.model == "mlp"):
            self.clf.fit(xTrain, yTrain, epochs = epochs, batch_size=16, validation_data=(X_validate, y_validate))
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
    """
    class to manage the data
    """
    def __init__(self, features, labels, chNum, featureNum):
        self.features = features
        self.labels = labels
        self.chNum = chNum
        self.featureNum = featureNum
        self.clasNum = max(labels["mov"])+1
         
    def getData(self):
        return self.features, self.labels
    
    def getSplitData(self, dropZeroSize = 9/10, testSize = 1/7):        
        a, b = train_test_split(np.where(self.labels["mov"]==0)[0], test_size=dropZeroSize)
        self.features.drop(a, inplace=True)
        self.labels.drop(a, inplace=True)    
        index = np.arange(self.labels.shape[0])
        self.features.index = index
        self.labels.index = index
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size = testSize)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=testSize)
        return X_train, X_validate, X_test, y_train, y_validate, y_test
        
   
        