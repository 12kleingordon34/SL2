# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:47:14 2019

@author: dantr
"""
import numpy as np

class LinearRegression(object):
    """Linear regression classifier"""
    def __init__(self):
        self.w = 0 # weights
        
    def train(self,X,y):
        """Train function
            Uses pseudo-inverse to allow for underdetermined case
        """
        self.w = np.linalg.pinv(X.T@X)@X.T@y
        
    def _predict(self,x):
        """Prediction (continuous)"""
        return(x@self.w)
        
    def predict(self,x):
        """Classifier"""
        yhat = self._predict(x)
        return(np.sign(yhat))
