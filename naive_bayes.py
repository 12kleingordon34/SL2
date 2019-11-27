# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:49:57 2019

@author: dantr
"""
import numpy as np

class NaiveBayes(object):
    def __init__(self):
        
        self.mean = 0
        self.cov = 0
        self.class_counts = 0
        self.labels = {}
        self.num_classes = 1
        
    def train(self,X,y):
        m,d = X.shape
        self.labels = list(set(y)).sort
        self.num_classes = len(self.labels)
        self.mean = np.zeros((self.num_classes,d))
        self.cov = np.zeros((self.num_classes,d,d))
        self.class_counts = np.zeros(self.num_classes)
        
        for i,j in enumerate(self.labels):
            self.class_counts[i] = np.sum(y==j)
            self.mean[i] = np.mean(X[y==j],axis=0)
            self.cov[i] = np.cov(X[y==j],axis=0)
        
        
    def _gaussian(self,x, mean ,cov):
        
        d = np.mean.size
        invvar = np.linalg.inv(cov)
        centered = x-mean.reshape(-1,1)
        coef = ((2*np.pi)**d *np.linalg.det(cov))**-0.5
        expon = np.exp(-0.5* np.sum((centered.T@invvar).T*centered,axis=0))
        return(coef*expon)
        
        
    def predict_proba(self,x):
        probs = np.zeros((x.shape[0],self.num_classes))
        
        for i in range(self.num_classes):
            p_class = self.class_counts[i]/np.sum(self.class_counts)
            probs[i] = self._gaussian(x,self.mean[i],self.cov[i])*p_class
        
        return(probs)
        
    def predict(self,x):
        probs = self.predict_proba(x)
        ind = np.argmax(probs,axis=1)
        return(self.labels[ind])

