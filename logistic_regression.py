# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:13:10 2019

@author: dantr
"""
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
    """Multinomial logistic regression class"""
    
    def __init__(self,lr = None,reg=0):
        self.w = 0 # weight matrix
        self.w_grad =0 # gradient matrix
        self.labels=0 # actual labels for reference
        self.num_labels = 0 # number of unique labels
        self.lr = lr # learning rate
        self.reg = reg # regularisation parameter
        self.b = 1 # bias term
        
        
    def train(self,X,y,mini_batch=5,num_iter = 10000,tol=0.01):
        """Train function
            X: data
            y: labels (single value)
            mini_batch: number of mini_batches
            num_iter: max iterations
            tol: stopping creterion
        """
        n,m = X.shape
        if self.lr is None:
            self.lr = 1
        
        if mini_batch == 0:
            mini_batch= 1
        elif mini_batch is None or mini_batch == "max":
            mini_batch = n
        
        y_o = self._one_hot_encode(y) # labels to vector
        # initialise weights
        self.w = np.zeros((m,self.num_labels))
        self.b = np.ones((self.num_labels))
        cost = np.zeros(num_iter)
        for i in range(num_iter):
            for j in range(mini_batch):
                i1 = int(j*n/mini_batch)
                i2 = min(int((j+1)*n/mini_batch),n)
                self._GD(X[i1:i2,:],y_o[i1:i2,:]) # gradient descent step
            cost[i] = self._cost(X,y_o)
            if i>2:
                # Half learning rate if subsequent costs increase
                if cost[i-1] == min([cost[i-1], cost[i]]):
                    self.lr = self.lr/2
                
            if i%500 ==0:
                self.lr = self.lr/(1+0.1)
                #print("Time: {}, Cost: {}".format(datetime.now(), self._cost(X,y_o)))
            # Relative Gradient Descent
            if abs((cost[i]-cost[i-1])/cost[i])< tol:
                break
        self.lr = 0.01
        for k in range(i+1,i+3): # finish with 2 full GD steps
            self._GD(X,y_o)
            cost[k] = self._cost(X,y_o)
        #print("Cost: {}".format(self._cost(X,y_o)))
        plt.plot(cost[:k])

        
    def predict_proba(self,x):
        """Predicted probabilites for new data x"""
        return(self.softmax(x))
        

    def predict(self,x):
        """Predict labels for new data x"""
        return(self.labels[np.argmax(self.predict_proba(x),axis=1)])
        
        
    def softmax(self,Y):
        """Softmax function
           Adjusts exponent by max for numerical stability 
        """
        scores = np.dot(Y,self.w) + self.b
        m = np.max(scores,axis=1,keepdims=True)
        num = np.exp(scores-m)
        denom = np.sum(num,axis=1,keepdims=True)
        return(num/denom)
        

    def _GD(self,x,y):
        """
        Gradient descent step
        """
        yhat = self.softmax(x) #predicted labels
        diff = yhat - y # difference
        step = np.dot(x.T,diff)/x.shape[0] # gradient
        self.b = self.b - self.lr*(np.mean(diff,axis=0)) # update bias (no regularisation)
        self.w = self.w - self.lr*(step - self.reg*self.w) # update weights
        

    def _cross_entropy(self,yhat,y):
        """Categorical Cross Entropy"""
        return(-np.sum(y*np.log(yhat+1e-8),axis=1))

        
    def _cost(self,x,y):
        """ Cost/Loss function  = Cross entropy + regularisation
        """
        yhat = self.softmax(x)
        cross = self._cross_entropy(yhat,y)
        cost = np.sum(cross) + 0.5 *self.reg * np.sum(self.w * self.w)
        return cost
    
    def _one_hot_encode(self,y):
        """ Transforms categorical labels to one-hot encode vectors.
            Updates labels and num_labels attributes
        """
        self.labels = np.array(list(set(y)))
        self.num_labels = self.labels.size
        y_one_hot = np.zeros((y.size,self.num_labels))
        for i,j in enumerate(self.labels):
            y_one_hot[:,i] = 1* (y==j)
        return(y_one_hot)
