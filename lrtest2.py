# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:13:10 2019

@author: dantr
"""
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
    
    def __init__(self,lr = None,reg=0):
        self.w = 0
        self.w_grad =0
        self.labels=0
        self.num_labels = 0
        self.lr = lr
        self.reg = reg
        self.b = 1
        
    def train(self,X,y):
        
        n,m = X.shape
        if self.lr is None:
            self.lr = 1/(1*n)
        
        
        y_o = self._one_hot_encode(y)#[:,:-1]
        
        self.w = np.zeros((m,self.num_labels))
        #self.b = np.ones((self.num_labels))
        cost = np.zeros(1000)
        for i in range(1000):
            
            self._GD(X,y_o)
            cost[i] = self._cost(X,y_o)
            if i%100 ==0:
                print("Cost: {}".format(self._cost(X,y_o)))
        
        plt.plot(cost)
        
    def predict_proba(self,x):
        out= self.softmax(x)
        
        return(out)
        
    def predict(self,x):
        
        return(self.labels[np.argmax(self.predict_proba(x),axis=1)])
        
        
    def softmax(self,Y):
        scores = np.dot(Y,self.w) + self.b
        m = np.max(scores)
        num = np.exp(scores-m)
        denom = np.sum(num,axis=1,keepdims=True)#1
        return(num/denom)
        
        
    def _GD(self,x,y):
        #for i in range(self.w.shape[1]):
        yhat = self.softmax(x)
        diff = yhat - y
        #print(np.sum(diff))
        step = np.dot(x.T,diff)/x.shape[0]
        #print(step)
        #self.b -= np.sum(diff,axis=0)
        self.w -= self.lr*(step - self.reg*self.w)
        #print(self.w)
        
    def _GD2(self,x,y):
        
        yhat = self.softmax(x)
        #print(yhat.shape)
        diff = yhat-y
        
        step = np.zeros((x.shape[1],self.num_labels))
        #print(step.shape)
        for i in range(self.num_labels):
            step[:,i] = np.mean(x*diff[:,i,None],axis=0)
        
        #print(step)
        #self.b = self.b - np.sum(diff,axis=0)
        self.w = self.w -  self.lr*(step - self.reg*self.w)    
        
        
        
        
    def _cross_entropy(self,yhat,y):
        
        return(-np.sum(y*np.log(yhat),axis=1)) #+ 0.5 *self.reg * np.sum(np.power(self.w,2)))
        
    def _cost(self,x,y):
        
        yhat = self.softmax(x)
        cross = self._cross_entropy(yhat,y)
        
        return(np.mean(cross) + 0.5 *self.reg * np.sum(self.w * self.w))
    
    def _one_hot_encode(self,y):
        self.labels = np.array(list(set(y)))
        self.num_labels = self.labels.size
        y_one_hot = np.zeros((y.size,self.num_labels))
        for i,j in enumerate(self.labels):
            y_one_hot[:,i] = 1* (y==j)
        
        return(y_one_hot)

