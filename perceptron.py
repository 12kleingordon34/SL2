# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:18:03 2019

@author: ucabdbt
"""
import numpy as np


class Perceptron(object):
    def __init__(self):
        self.w = np.array([])
        self.num_classes = 1
        self.M = 0
        

    def train(self,X,y,num_epochs=1):
        m,d = X.shape
        self.num_classes = np.atleast_2d(y).shape[0]
        self.M = np.zeros(self.num_classes)
        
        self.w = np.zeros((d,self.num_classes))
        for e in range(num_epochs):
            self._training_run(X, y, m)

        
    def _training_run(self, X, y, m):
        for t in range(m):
            yhat = np.sign(np.dot(X[t,:],self.w))
            if yhat*y[t] <= 0:
                self.w += y[t]*X[t,:,None]
                self.M += 1
                

    def predict_proba(self,X):
        if self.w.shape[1]==1:
           return(np.dot(X,self.w).flatten())
        return(np.dot(X,self.w))
        

    def predict(self,X):
        return(np.sign(self.predict_proba(X)))


class KernelPerceptron(object):
    def __init__(self,kernel,k_params):
        self.w = np.array([])
        self.num_classes = 1
        self.M = 0
        self.train_set = 0
        self.k_params = k_params
        self.kernel = kernel
        

    def build_gram(self,X):
        return(self.kernel(X,X,self.k_params))


    def train(self, X, y, num_epochs=1):
        self.train_set = X
        m,d = X.shape
        self.num_classes = np.atleast_2d(y).shape[0]
        self.M = np.zeros(self.num_classes)
        gram = self.build_gram(X)
        self.w = np.zeros(m)

        for e in range(num_epochs):
            self._training_run(y, gram, m)

        
    def _training_run(self, y, gram, m):
        for i in range(m):
            if np.sign(np.dot(self.w,gram[:,i])) != y[i]:
                self.w[i] += y[i]
                self.M+=1
    

    def predict_proba(self,x):
        k = self.kernel(self.train_set,x,self.k_params)
        return(np.dot(self.w,k))
        

    def predict(self,x):
        return(np.sign(self.predict_proba(x)))
