# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:19:46 2019

@author: dantr
"""

import numpy as np

class OneNN(object):
    
    def __init__(self,X,y):
        self.x = X
        self.y = y
        
        
    def predict(self,X):
        
        dist = self._build_dist(X)
        idx = np.argmin(dist,axis=0)
        
        return(self.y[idx])
        
    def _build_dist(self,X):
        l1,n1 = self.x.shape
        l2,n2 = X.shape
        assert n1==n2
    
        K = np.zeros((l1,l2))
        for i in range(l1):
            K[i,:] = self._build_dist_row(X[i,:])
            return(K)
        
    def _build_dist_row(self,X):
        return  np.sum(np.power(self.x-X,2),axis=1)
        