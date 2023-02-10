# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:56:53 2022

@author: Renato
"""

import numpy as np
import math
import lmoments3 as lm
import pywt
import clases as cl
from sklearn.model_selection import cross_validate

def mediana(data):
    out = data.copy()
    for i in range(len(data)-3):
        out[i] = np.median(data[i:i+3])
    return out


def secuentialChannelSelection(xTrain, yTrain, xTest, yTest, nFeatures, maxChannels, clasNum, model):
    channelSelect = []
    accuracyList = []
    allChannels = np.array(xTrain.columns)
    for n in range(maxChannels):
        accuracy = 0
        j = 0
        red = cl.Clasificador(model= model, catNum = clasNum, featureNum = len(channelSelect)+nFeatures)
        for i in range(int(len(allChannels)/nFeatures)):
            print("Step: ", n, " Channel:", i)
            index = np.arange(nFeatures)+i*nFeatures
            columns = np.concatenate((channelSelect, allChannels[index]))
            red.train(xTrain[columns], yTrain)
            yPredict = red.predict(xTest[columns])
            print("Accuracy: ",np.mean(yPredict==yTest) )        
            if(np.mean(yPredict==yTest)>accuracy):
                accuracy = np.mean(yPredict==yTest)
                j = i
        index = np.arange(nFeatures)+j
        channelSelect = np.concatenate((channelSelect, allChannels[index]))
        allChannels = np.delete(allChannels, index)
        accuracyList.append(accuracy)
    return channelSelect, accuracyList

def accuracyByChannel(xTrain, yTrain, xTest, yTest, nFeatures, clasNum, model):
    channelSelect = []
    accuracyList = []
    allChannels = np.array(xTrain.columns)
    n = 0
    red = cl.Clasificador(model= model, catNum = clasNum, featureNum = nFeatures)
    for i in range(int(len(allChannels)/nFeatures)):
        print("Step: ", n, " Channel:", i)
        index = np.arange(nFeatures)+i*nFeatures
        columns = np.concatenate((channelSelect, allChannels[index]))
        red.train(xTrain[columns], yTrain)
        yPredict = red.predict(xTest[columns])
        accuracy = np.mean(yPredict==yTest)
        print("Accuracy: ",accuracy )     
        accuracyList.append(accuracy)
    return channelSelect, accuracyList

def secuentialFeatureSelection(xTrain, yTrain, xTest, yTest, nFeatures, maxFeatures, nChannels, clasNum, model):
    featureSelect = []
    accuracyList = []
    allFeatures = np.array(xTrain.columns)
    for n in range(nFeatures):
        accuracy = 0
        j = 0
        red = cl.Clasificador(model= model, catNum = clasNum, featureNum = len(featureSelect)+nChannels)
        for i in range(int(len(allFeatures)/nChannels)):
            index = np.arange(nChannels)*nFeatures+i
            columns = np.concatenate((featureSelect, allFeatures[index]))
            print("Step: ", n, " Feature: ", allFeatures[index[0]].split("_c")[0])
            red.train(xTrain[columns], yTrain)
            yPredict = red.predict(xTest[columns])
            print("Accuracy: ",np.mean(yPredict==yTest) )        
            if(np.mean(yPredict==yTest)>accuracy):
                accuracy = np.mean(yPredict==yTest)
                j = i
        index = np.arange(nChannels)*nFeatures+j
        nFeatures = nFeatures -1
        featureSelect = np.concatenate((featureSelect, allFeatures[index]))
        allFeatures = np.delete(allFeatures, index)
        accuracyList.append(accuracy)
    return featureSelect, accuracyList

def accuracyByFeature(xTrain, yTrain, xTest, yTest, nFeatures, nChannels, clasNum, model):
    featureList = []
    accuracyList = []
    allFeatures = np.array(xTrain.columns)
    red = cl.Clasificador(model= model, catNum = clasNum, featureNum = nChannels)
    for i in range(int(len(allFeatures)/nChannels)):
        index = np.arange(nChannels)*nFeatures+i
        columns = allFeatures[index]
        print(" Feature: ", allFeatures[index[0]].split("_c")[0])
        featureList.append(allFeatures[index[0]].split("_c")[0])
        red.train(xTrain[columns], yTrain)
        yPredict = red.predict(xTest[columns])
        accuracyList.append(np.mean(yPredict==yTest)*100)
        print("Accuracy: ",np.mean(yPredict==yTest) ) 
    return featureList, accuracyList


def sub_cat(data, label, cat, tiempo):
    sub_label = label[np.where(label==cat)[0]]
    sub_data = data.iloc[np.where(label==cat)[0]]
    return sub_data, sub_label
# =============================================================================
# Características utilizadas en "Classification of 41 Hand and Wrist"
def DWT(data):
    cA2, cD2, cD1 = pywt.wavedec(data, 'db7', mode='periodization', level=2)
    cA2 = np.sum(abs(cA2))
    cD2 = np.sum(abs(cD2))
    cD1 = np.sum(abs(cD1))
    return cA2, cD2, cD1

def rms(data):
    """
    Root Mean Square.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: RMS feature.
    """
    return math.sqrt(np.power(data, 2).sum() / data.size)

def mav(data):
    """
    Mean Amplitude Value.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: MAV feature.
    """
    return np.absolute(data).sum() / data.size

def mavs(data):
    """
    Mean Absolute Value Slope.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: MAVS feature.
    """
    return mav(data[1:])-mav(data[:-1])

def zc(data, threshold):
    """
    Zero Crossing.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: ZC feature.
    """
    return (data[:-1] * data[1:] < 0).sum()

def ssc(data, threshold ):
    """
    Slope Sign Change.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: SSC feature.
    """
    
    return ((data[1:-1] - data[:-2]) * (data[1:-1] - data[2:]) >= threshold).sum()

def wl(data):
    """
    Wavelength.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: WL feature.
    """
    
    return np.abs(data[1:] - data[:-1]).sum()
# =============================================================================
# =============================================================================
# Características utilizadas en "Prótesis de antebrazo con control mioeléctrico" Incluye las características anteriores
def ls(data):
    """
    L-Scale.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: LS feature.
    """
    return lm.lmom_ratios(data, nmom=2)[1]

def mfl(data):
    """
    Maximum Fractal Lenght.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: MFL feature.
    """
    return math.log10(math.sqrt(
        np.power((data[1:]-data[:-1]), 2).sum()
    ))

def msr(data):
    """
    Mean Square Root:

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: MSR feature.
    """
    return np.sqrt(np.abs(data)).sum() / data.size

def wamp(data):
    """
    Willison Amplitude.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: WAMP feature.
    """
    threshold = 0.01
    return (np.abs(data[:-1] - data[1:]) >= threshold).sum()


def iav(data):
    """
    Integral Absolute Value.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: IAV feature.
    """
    return np.trapz(np.abs(data))

def dasdv(data):
    """
    Difference Absolute Standard Deviation Value.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: DASDV feature.
    """
    return math.sqrt(
        np.power((data[1:] - data[:-1]), 2).sum() / 
        (data.size - 1)
    )

def _var(data):
    """
    Variance.

    Arguments:
        @var data: list of graph data.

    Returns:
        @return: VAR feature.
    """
    return np.power(data, 2).sum() / (data.size - 1)
# =============================================================================


