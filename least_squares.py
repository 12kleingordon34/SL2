# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:47:14 2019

@author: dantr
"""
import numpy as np

class LinearRegression(object):
    
    def __init__(self):
        
        self.w = 0
        
    def train(self,X,y):
        
        self.w = np.linalg.solve(X.T@X,X.T@y)
        
    def predict(self,x):
        
        return(x@self.w)
        
    def classify(self,x):
        
        yhat = self.predict(x)
        
        return(np.sign(yhat))
        
        