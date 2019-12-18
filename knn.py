# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:19:46 2019

@author: dantr
"""

import numpy as np
from scipy import spatial

class OneNN(object):
    
    def __init__(self):
        self.x = 0
        self.y = 0
        
    def train(self,X,y):
        self.x = X
        self.y = y
        
#    def predict(self,X):
#        dist = self._build_dist(X)
#        idx = np.argmin(dist,axis=1)
#        return(self.y[idx])
        
    def _build_dist(self,X):
        l1,n1 = self.x.shape
        l2,n2 = X.shape
        assert n1==n2
    
        K = np.zeros((l2,l1))
        for i in range(l2):
            K[i,:] = self._build_dist_row(X[i,:])
        return(K)
    
    def predict(self,X):
        l1,n1 = X.shape
        idx = np.zeros(l1)
        for i in range(l1):
            idx[i] = np.argmin(self._build_dist_row(X[i,:]))
        return(self.y[idx.astype(int)])

    
    def _build_dist_row(self,X):
        return  np.sum(np.power(self.x-X,2),axis=1)
        

class kNN(object):
    def __init__(self):
        self.x= None
        self.y = None
        self.tree = None

    def train(self, X, y):
        self.x = X
        self.y = y
        self.tree = spatial.KDTree(X)

    def predict(self, X_test, k=1):
        assert (type(k) == int) and (k > 0)

        # Find indexes and distances of nearest neighbours to test points
        dist, ind = self.tree.query(X_test, k=k)
        # Create Minkowski weights
        weights = 1/dist
        len_test = X_test.shape[0]
        y_pred = np.zeros(len_test)
        # Classify each row in test set
        for row in range(len_test):
            unique_classes = np.unique(self.y[ind][row])
            c_optimal = 0
            opt_weight = 0
            # Find class with the greatest sum(weights*c_counts)
            for c in unique_classes:
                if k == 1:
                    weight = weights[row]
                    if weight > opt_weight:
                        opt_weight = weight
                        c_optimal = c
                else:
                    weight = sum(weights[row][self.y[ind[row]]==c])
                    if weight > opt_weight:
                        opt_weight = weight
                        c_optimal = c
            y_pred[row] = c_optimal
        return y_pred
