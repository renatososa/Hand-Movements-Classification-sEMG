# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:28:33 2022
Script for convert .mat subject files to dict structure and export as a pickle.
Download the data from: http://ninapro.hevs.ch/data5, unzip it in the data folder.
Eg.: data/S1/S1_E1_A1.mat
Eg.: data/S2/S2_E1_A1.mat
Project: Study and prototyping of an automatic classification system for hand gestures using electromyography.
@author: Renato Sosa Machado Scheeffer. Universidad de la República.
"""

import scipy.io
import pickle
import numpy as np

nSubjects = 10
files = ['S' + str(x) for x in range(1,nSubjects+1)] 
data = dict()
for file in files:
    E1 = scipy.io.loadmat("../data/" + file + '/' + file + '_E1_A1.mat') # Exercise A
    E2 = scipy.io.loadmat("../data/" + file + '/' + file + '_E2_A1.mat') # Exercise B
    E3 = scipy.io.loadmat("../data/" + file + '/' + file + '_E3_A1.mat') # Exercise C
    
    E2["stimulus"][np.where(E2["stimulus"]==0)] = E2["stimulus"][np.where(E2["stimulus"]==0)]-12
    E3["stimulus"][np.where(E3["stimulus"]==0)] = E3["stimulus"][np.where(E3["stimulus"]==0)]-29
    
    data[file] = {'emg':np.concatenate([E1["emg"], E2['emg'], E3['emg']]),
                  'label':np.concatenate([E1["stimulus"], E2["stimulus"]+12, 
                E3["stimulus"] +29]), 'info':{'age':E1['age'], 'gender':E1['gender'], 
                          'n_subjects':10, 'weight':E1['weight'], 
                          'height':E1['height'], 'fs':E1["frequency"][0][0]}, 
                  'description':"Contiene datos de EMG de 10 sujetos, 3 experimentos por cada sujeto (E1, E2 y E3) (A, B y C de la figura), cada experimento contiene diferentes ejercicios (A:12, B:17, C:23)."}

with open("../data/MyoArmband_data.pickle", "wb") as f:
    pickle.dump(data, f)
    
    
