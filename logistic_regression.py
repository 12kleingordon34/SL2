# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:13:10 2019

@author: dantr
"""
import numpy as np

class LogisticRegression(object):
    
    def __init__(self):
        self.w = 0
        self.w_grad =0
        self.labels=0
        self.num_labels = 0
        self.lr = None
        self.reg = 0.1
        
    def train(self,X,y):
        
        
        
        n,m = X.shape
        if self.lr is None:
            self.lr = 1/n
        
        
        y_o = self._one_hot_encode(y)
        
        self.w = np.random.normal(0,1,(m,self.num_labels))
        
        for i in range(100):
            
            self._GD(X,y_o)
            
            print("Cost: {}".format(self._cost(X,y_o)))
            
        
        
    def predict_proba(self,x):
        return(self.softmax(x))
        
    def predict(self,x):
        
        return(self.labels[np.argmax(self.predict_proba(x),axis=1)])
        
        
    def softmax(self,Y):
        scores = np.dot(Y,self.w)
        m = np.max(scores)
        num = np.exp(scores-m)
        denom = np.sum(num,axis=1,keepdims=True)
        return(num/denom)
        
        
    def _GD(self,x,y):
        #for i in range(self.w.shape[1]):
        yhat = self.softmax(x)
        diff = yhat - y
        #print(diff)
        step = np.dot(x.T,diff)/x.shape[0]
        #print(step)
        self.w -= self.lr*(step - self.reg*self.w)
        #print(self.w)
        
    def _cross_entropy(self,yhat,y):
        
        return(-np.sum(y*np.log(yhat+1e-6),axis=1)) #+ 0.5 *self.reg * np.sum(np.power(self.w,2)))
        
    def _cost(self,x,y):
        
        yhat = self.softmax(x)
        cross = self._cross_entropy(yhat,y)
        
        return(np.mean(cross) + 0.5 *self.reg * np.sum(np.power(self.w,2)))
    
    def _one_hot_encode(self,y):
        self.labels = np.array(list(set(y)))
        self.num_labels = self.labels.size
        y_one_hot = np.zeros((y.size,self.num_labels))
        for i,j in enumerate(self.labels):
            y_one_hot[:,i] = 1* (y==j)
        
        return(y_one_hot)