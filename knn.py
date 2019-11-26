# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:19:46 2019

@author: dantr
"""

import numpy as np

class OneNN(object):
    
    def __init__(self,X,y):
        self.x = X
        self.y = y
        
        
    def predict(self,X):
        
        dist = self.x @ X.T
        idx = np.argmin(dist,axis=0)
        
        return(self.y[idx])
        