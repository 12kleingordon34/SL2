# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:02:50 2019

@author: dantr
"""
import numpy as np

class Winnow(object):
    
    def __init__(self):
        self.w = 1
        self.M = 0
        
        
    def train(self,X,y):
        
        m,n = X.shape
        
        self.w = np.ones(n)
        
        for i in range(m):
            
            yhat = 1* (np.dot(self.w,X[i,:]) >= n)
            
            if yhat != y[i]:
                self.w = self.w * np.power(2, (y[i]-yhat)*X[i,:])
                self.M += 1
                
    def predict_proba(self,x):
        
        return(x@self.w)
        
    def predict(self,x):
        
        return(1*(self.predict_proba(x) >= w.size)