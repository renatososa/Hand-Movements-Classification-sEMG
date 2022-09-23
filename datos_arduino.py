# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:19:14 2022

@author: Renato
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("prueba.txt")
plt.figure()
plt.plot(np.arange(len(data))/790,(data-1500)/1000)