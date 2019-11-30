# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:07:41 2019

@author: ucabdbt
"""

def loss_grad_softmax_vectorized(W, X, y, reg):
    """ Compute the loss and gradients using softmax with vectorized version"""
    loss = 0 
    grad = np.zeros_like(W)
    dim, num_train = X.shape

    scores = W.dot(X) # [K, N]
    # Shift scores so that the highest value is 0
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    correct_scores_exp = scores_exp[y, range(num_train)] # [N, ]
    scores_exp_sum = np.sum(scores_exp, axis=0) # [N, ]
    loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    scores_exp_normalized = scores_exp / scores_exp_sum
    # deal with the correct class
    scores_exp_normalized[y, range(num_train)] -= 1 # [K, N]
    grad = scores_exp_normalized.dot(X.T)
    grad /= num_train
    grad += reg * W

    return loss, grad


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:13:10 2019

@author: dantr
"""
import numpy as np

class LogisticRegression(object):
    
    def __init__(self,lr = None):
        self.w = 0
        self.w_grad =0
        self.labels=0
        self.num_labels = 0
        self.lr = lr
        self.reg = 0.1
        #self.b = 1
        self.B = 0
        self.Binv =0
        self.BinvB = 0
        
    def train(self,X,y):
        
        
        
        n,m = X.shape
        
        
        
        y_o = self._one_hot_encode(y)
        
        self.w = np.random.normal(0,1,(m*(self.num_labels-1)))
        #self.b = np.ones((self.num_labels))
        
        l = np.eye(self.num_labels-1) -np.ones((self.num_labels-1,self.num_labels-1))/self.num_labels
        r = np.dot(X.T,X)
        
        self.B = -0.5* np.kron(l,r)
        self.Binv = np.linalg.inv(self.B - self.reg*np.eye(m*(self.num_labels-1)))
        self.BinvB = np.dot(self.Binv,self.B)
        
        for i in range(10):
            
            self._GD(X,y_o)
            if i%100 ==0:
                print("Cost: {}".format(self._cost(X,y_o)))
        print("Cost: {}".format(self._cost(X,y_o)))
            
        
        
    def predict_proba(self,x):
        return(self.softmax(x))
        
    def predict(self,x):
        
        return(self.labels[np.argmax(self.predict_proba(x),axis=1)])
        
        
    def softmax(self,Y):
        w = self.w.reshape(self.num_labels-1,Y.shape[1]).T
        scores = np.dot(Y,w) #+ self.b
        m = np.max(scores)
        num = np.exp(scores-m)
        denom = np.sum(num,axis=1,keepdims=True)+1
        return(num/denom)
        
        
    def _GD(self,x,y):
        #for i in range(self.w.shape[1]):
        yhat = self.softmax(x)
        diff = y[:,:-1] - yhat
        #print(diff)
        grad = np.sum(np.kron(diff.T,x.T),axis=1)
        #print(step)
        #self.b += np.sum(diff,axis=0)
        self.w = self.BinvB@self.w -  np.dot(self.Binv,grad)
        #print(self.w)
        
    def _cross_entropy(self,yhat,y):
        
        return(-np.sum(y[:,:-1]*np.log(yhat+1e-6),axis=1)) #+ 0.5 *self.reg * np.sum(np.power(self.w,2)))
        
    def _cost(self,x,y):
        
        yhat = self.softmax(x)
        cross = self._cross_entropy(yhat,y)
        
        return(np.mean(cross)) #+ 0.5 *self.reg * np.sum(np.power(self.w,2)))
    
    def _one_hot_encode(self,y):
        self.labels = np.array(list(set(y)))
        self.num_labels = self.labels.size
        y_one_hot = np.zeros((y.size,self.num_labels))
        for i,j in enumerate(self.labels):
            y_one_hot[:,i] = 1* (y==j)
        
        return(y_one_hot)