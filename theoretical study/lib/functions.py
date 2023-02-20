# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:56:53 2022
In this library the necessary functions for the project are defined.
Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""
import numpy as np
import math
import lmoments3 as lm
import pywt
import clases as cl

def secuentialChannelSelection(xTrain, yTrain, xValidate, yValidate, xTest, yTest, nFeatures, maxChannels, clasNum, model):
    """
    Function to calculate a sequential channel selectrion
    Parameters
    ----------
    xTrain : dataFrame
        Features to he training.
    yTrain : dataFrame
        labels to the training.
    xValidate : dataFrame
        Features to the validation.
    yValidate : TYPE
        Labels to the validation.
    xTest : dataFrame
        Features to the test.
    yTest : dataFrame
        Labels to the test.
    nFeatures : int
        Number of features per channel.
    maxChannels : int
        max number of channels in the secuenctial selection.
    clasNum : int
        number of classes in the clasification.
    model : str
        name of the selected model.

    Returns
    -------
    channelSelect : list of str list
        list with the best selected channels.
    accuracyList : list of int list 
        list of accuracies corresponding to the best selected channels.
    """
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
            red.train(xTrain[columns], yTrain, xValidate[columns], yValidate)
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

def accuracyByChannel(xTrain, yTrain, xValidate, yValidate, xTest, yTest, nFeatures, clasNum, model):
    """
    Parameters
    ----------
    xTrain : dataFrame
        Features to he training.
    yTrain : dataFrame
        labels to the training.
    xValidate : dataFrame
        Features to the validation.
    yValidate : TYPE
        Labels to the validation.
    xTest : dataFrame
        Features to the test.
    yTest : dataFrame
        Labels to the test.
    nFeatures : int
        Number of features per channel.
    clasNum : int
        number of classes in the clasification.
    model : str
        name of the selected model.

    Returns
    -------
    channel : str list
        channels name.
    accuracyList : int list
        channels acccuracy.

    """
    channel = []
    accuracyList = []
    allChannels = np.array(xTrain.columns)
    n = 0
    red = cl.Clasificador(model= model, catNum = clasNum, featureNum = nFeatures)
    for i in range(int(len(allChannels)/nFeatures)):
        print("Step: ", n, " Channel:", i)
        index = np.arange(nFeatures)+i*nFeatures
        columns = np.concatenate((channel, allChannels[index]))
        red.train(xTrain[columns], yTrain, xValidate[columns], yValidate)
        yPredict = red.predict(xTest[columns])
        accuracy = np.mean(yPredict==yTest)
        print("Accuracy: ",accuracy )     
        accuracyList.append(accuracy)
    return channel, accuracyList

def secuentialFeatureSelection(xTrain, yTrain, xValidate, yValidate, xTest, yTest, nFeatures, maxFeatures, nChannels, clasNum, model):
    """
    Function to calculate a sequential feature selectrion
    Parameters
    ----------
    xTrain : dataFrame
        Features to he training.
    yTrain : dataFrame
        labels to the training.
    xValidate : dataFrame
        Features to the validation.
    yValidate : TYPE
        Labels to the validation.
    xTest : dataFrame
        Features to the test.
    yTest : dataFrame
        Labels to the test.
    nChannels : int
        Number of channels per channel.
    maxFeatures : int
        max number of features in the secuenctial selection.
    clasNum : int
        number of classes in the clasification.
    model : str
        name of the selected model.

    Returns
    -------
    featureSelect : list of str list
        list with the best selected features.
    accuracyList : list of int list 
        list of accuracies corresponding to the best selected features.
    """
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
            red.train(xTrain[columns], yTrain, xValidate[columns], yValidate)
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

def accuracyByFeature(xTrain, yTrain, xValidate, yValidate, xTest, yTest, nFeatures, nChannels, clasNum, model):
    """
    Parameters
    ----------
    xTrain : dataFrame
        Features to he training.
    yTrain : dataFrame
        labels to the training.
    xValidate : dataFrame
        Features to the validation.
    yValidate : TYPE
        Labels to the validation.
    xTest : dataFrame
        Features to the test.
    yTest : dataFrame
        Labels to the test.
    nFeatures : int
        Number of features per channel.
    nChannels : int
        Number of channels.   
    clasNum : int
        number of classes in the clasification.
    model : str
        name of the selected model.

    Returns
    -------
    feature : str list
        channels name.
    accuracyList : int list
        channels acccuracy.

    """
    feature = []
    accuracyList = []
    allFeatures = np.array(xTrain.columns)
    red = cl.Clasificador(model= model, catNum = clasNum, featureNum = nChannels)
    for i in range(int(len(allFeatures)/nChannels)):
        index = np.arange(nChannels)*nFeatures+i
        columns = allFeatures[index]
        print(" Feature: ", allFeatures[index[0]].split("_c")[0])
        feature.append(allFeatures[index[0]].split("_c")[0])
        red.train(xTrain[columns], yTrain, xValidate[columns], yValidate)
        yPredict = red.predict(xTest[columns])
        accuracyList.append(np.mean(yPredict==yTest)*100)
        print("Accuracy: ",np.mean(yPredict==yTest) ) 
    return feature, accuracyList


def sub_cat(data, label, cat):
    """
    Select a specific label form data

    Parameters
    ----------
    data : dataFrame
        features.
    label : array
        labels.
    cat : int
        label to extract.

    Returns
    -------
    sub_data : dataFrame
        datFrame correspond to the slected label.
    sub_label : array
        labels.

    """
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


