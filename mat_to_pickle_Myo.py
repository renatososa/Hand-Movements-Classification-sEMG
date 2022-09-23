# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:28:33 2022

@author: Renato
"""

import scipy.io
import pickle
import numpy as np

files = ['S' + str(x) for x in range(1,11)]
data = dict()
for file in files:
    E1 = scipy.io.loadmat(file + '/' + file + '_E1_A1.mat')
    E2 = scipy.io.loadmat(file + '/' + file + '_E2_A1.mat')
    E2["stimulus"][np.where(E2["stimulus"]==0)] = E2["stimulus"][np.where(E2["stimulus"]==0)]-12
    E3 = scipy.io.loadmat(file + '/' + file + '_E3_A1.mat')
    E3["stimulus"][np.where(E3["stimulus"]==0)] = E3["stimulus"][np.where(E3["stimulus"]==0)]-29
    
    data[file] = {'emg':np.concatenate([E1["emg"], E2['emg'], E3['emg']]),
                  'label':np.concatenate([E1["stimulus"], E2["stimulus"]+12, 
                E3["stimulus"] +29]), 'info':{'age':E1['age'], 'gender':E1['gender'], 
                          'n_subjects':10, 'weight':E1['weight'], 
                          'height':E1['height'], 'fs':E1["frequency"][0][0]}, 
                  'description':"Contiene datos de EMG de 10 sujetos, 3 experimentos por cada sujeto (E1, E2 y E3) (A, B y C de la figura), cada experimento contiene diferentes ejercicios (A:12, B:17, C:23)."}

with open("MyoArmband_data.pickle", "wb") as f:
    pickle.dump(data, f)
    
    
