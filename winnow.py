# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:02:50 2019

@author: dantr
"""
import numpy as np

class Winnow(object):
    """Winnow class"""
    def __init__(self):
        self.w = 1 # weights
        self.M = 0 # number of mistakes
        
    def train(self,X,y):
        """Train function
            X = data
            y = labels (0,1)
        """
        m,n = X.shape
        # initialise weights
        self.w = np.ones(n)       
        for i in range(m):
            # predict new data
            yhat = 1* (np.dot(self.w,X[i,:]) >= n)            
            if yhat != y[i]: # if mistake update weights
                self.w = self.w * np.power(2., (y[i]-yhat)*X[i,:])
                self.M += 1
                
    def predict_proba(self,x):
        """Returns margin for new data x"""
        return(x@self.w)
        
    def predict(self,x):
        """Predict class for new data x"""
        return(1*(self.predict_proba(x) >= self.w.size))