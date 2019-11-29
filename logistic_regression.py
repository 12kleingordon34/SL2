# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:13:10 2019

@author: dantr
"""
import numpy as np

class LogisticRegression(object):
    
    def __init__(self)
        self.w = 0
        self.w_grad =0
        self.labels=0
        
    def train(self,X,y):
        
    def predict_proba(self,x):
        return(self.softmax(np.dot(x,self.w)))
        
    def predict(self,x):
        
        
        
    def softmax(self,Y):
        m = np.max(Y)
        num = np.exp(Y-m)
        denom = np.sum(num)
        return(num/denom)
        
        
    def _SGD(self,lr):
        for i in range(self.w.size):
            self.w[i] -= lr*self.w_grad[i]
        
    def _cross_entropy(self,yhat,y):
        
        return(-np.sum(y*np.log(yhat)))
        
    def _one_hot_encode(self,y):
        self.labels = list(set(y))
        
        y_one_hot = np.array(y.size,len(self.labels))
        for i in self.labels:
            y_one_hot[:,i] = 1* y==self.labels[i]
        
        return(y_one_hot)