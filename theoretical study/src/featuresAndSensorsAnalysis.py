# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:41:46 2022
Script for sensor and feature analysis.
@Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""
import extraccion_caract as features
from joblib import load
import clases as cl
import pandas as pd
from joblib import dump

FEATURES = load("../data/FEATURES.joblib")
label = load("../data/label.joblib")

data = cl.data(features = FEATURES, labels = label, chNum = 16, featureNum = 15)

# Split de los datos
X_train, X_validate, X_test, y_train, y_validate, y_test, = data.getSplitData()
y_train = pd.Series(y_train["mov"], dtype="category")
y_test = pd.Series(y_test["mov"], dtype = "category")
y_validate = pd.Series(y_validate["mov"], dtype = "category")
nChannels = data.chNum
nFeatures = data.featureNum
inputDim = nChannels*nFeatures 
clasNum = data.clasNum

modelo = "mlp"
channels, accuracyList = features.secuentialChannelSelection(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nChannels, clasNum, model = modelo)
dump([channels, accuracyList], "Datos/accuracyPerChannelMLP.joblib")

channels, accuracyList = features.accuracyByChannel(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures,  clasNum = clasNum, model = modelo)
dump([channels, accuracyList], "Datos/accuracyByChannelMLP.joblib")

feature, accuracyList = features.secuentialFeatureSelection(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nFeatures, nChannels, clasNum , model = modelo)
dump([features, accuracyList], "Datos/accuracyPerFeaturelMLP.joblib")

feature, accuracyList = features.accuracyByFeature(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nChannels, clasNum, model = modelo)
dump([features, accuracyList], "Datos/accuracyByFeatureMLP.joblib")


modelo = "svm"
channels, accuracyList = features.secuentialChannelSelection(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nChannels, clasNum, model = modelo)
dump([channels, accuracyList], "Datos/accuracyPerChannelSVM.joblib")

channels, accuracyList = features.accuracyByChannel(X_train, y_train,
X_validate, y_validate, X_test, y_test, nFeatures,  clasNum = clasNum, model = modelo)
dump([channels, accuracyList], "Datos/accuracyByChannelSVM.joblib")

feature, accuracyList = features.secuentialFeatureSelection(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nFeatures, nChannels, clasNum , model = modelo)
dump([features, accuracyList], "Datos/accuracyPerFeaturelSVM.joblib")

feature, accuracyList = features.accuracyByFeature(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nChannels, clasNum, model = modelo)
dump([features, accuracyList], "Datos/accuracyByFeatureSVM.joblib")


modelo = "gbm"
channels, accuracyList = features.secuentialChannelSelection(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nChannels, clasNum , model = modelo)
dump([channels, accuracyList], "Datos/accuracyPerChannelGBM.joblib")

channels, accuracyList = features.accuracyByChannel(X_train, y_train,
X_validate, y_validate, X_test, y_test, nFeatures,  clasNum = clasNum , model = modelo)
dump([channels, accuracyList], "Datos/accuracyByChannelGBM.joblib")

feature, accuracyList = features.secuentialFeatureSelection(X_train, y_train,
X_validate, y_validate, X_test, y_test, nFeatures, nFeatures, nChannels, clasNum , model = modelo)
dump([features, accuracyList], "Datos/accuracyPerFeaturelGBM.joblib")

feature, accuracyList = features.accuracyByFeature(X_train, y_train, 
X_validate, y_validate, X_test, y_test, nFeatures, nChannels, clasNum , model = modelo)
dump([features, accuracyList], "Datos/accuracyByFeatureGBM.joblib")