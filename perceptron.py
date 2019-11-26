# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:18:03 2019

@author: ucabdbt
"""

import numpy as np

import kernels

class Perceptron(object):
    
    def __init__(self):
        self.w = np.array([])
        self.num_classes = 1
        self.M = 0
        self.R = 0
        
    def train(self,X,y):
        m,d = X.shape
        
        self.num_classes = y.shape[1]
        self.M = np.zeros(self.num_classes)
        self.R = np.max(np.sum(X*X,axis=1))
        
        self.w = np.zeros((d,self.num_classes))
        
        for t in range(m):
            yhat = np.sign(np.dot(X[t,:],self.w))
            if yhat*y[t,:] <= 0:
                self.w += y[t]*X[t,:]
                self.M += 1
                
    def predict_proba(self,X):
        return(np.dot(X,self.w))
        
    def predict(self,X):
        return(np.sign(self.predict_proba(self,X)))
