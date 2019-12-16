# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:45:07 2019

@author: dantr
"""
import numpy as np

def JALB(n,m,seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = -1 + 2*np.random.binomial(1,0.5,size=(m*n)).reshape(m,n)
    y = X[:,0]
    return(X,y)
