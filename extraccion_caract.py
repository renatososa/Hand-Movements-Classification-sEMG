# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:56:53 2022

@author: Renato
"""

import numpy as np
import math
import lmoments3 as lm
import pywt

from sklearn.model_selection import cross_validate

def mediana(data):
    out = data.copy()
    for i in range(len(data)-3):
        out[i] = np.median(data[i:i+3])
    return out


def re_label(labels, old_labels, new_label):
    for i in range(len(old_labels)):
        labels[np.where(labels == old_labels[i])[0]] = new_label
    return labels

def SFC(model, X, y, splits, n, by, N_features):
    features = []
    scores = np.zeros(n)
    all_scores = []
    times = np.zeros(n)
    info = []
    if(by=='channel'):
        all_features = list(X.columns)    
        N = N_features
    else:
        all_features.sort()
        N = 16
    for i in range(n):
        m_accuracy = 0
        for j in range(int(len(all_features)/N)):
            features_aux = features + all_features[N*j:(j+1)*N]
            cv_results = cross_validate(model, X[features_aux], y, cv=splits)
            accuracy = cv_results['test_score'].mean()*100
            time = cv_results['fit_time'].mean()
            if accuracy>m_accuracy:
                times[i] = time
                m_accuracy = accuracy
                select_features = all_features[N*j:(j+1)*N]
                ind = j
                all_scores
        del all_features[N*ind:(ind+1)*N]
        features = features + select_features
        select_features = []
        scores[i] = m_accuracy
        all_scores.append(cv_results['test_score'])
        info.append(all_features)
    return features, scores, times, info, all_scores

def sub_cat(data, label, cat):
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
    threshold = 50
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


