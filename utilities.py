# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:02:06 2019

@author: dantr
"""
import numpy as np


def train_test_split(X, y, percentage=1/3, seed=None):
    """ Splits a dataset into training and test sets
        with a random permutation
    
        Inputs:
            X: data matrix
            y: values
            percentage: proportion of data in test set
        Outputs:
            X_train,X_test,y_train,y_test: split data
    """
    l,n = X.shape
    # randomly permute the data
    if seed:
        np.random.seed(seed)
    perm = np.random.permutation(np.arange(l))
    # split data proportionally
    index = perm[:int((1-percentage)*l)]
    negindex = perm[int((1-percentage)*l):]
    X_train = X[index,:]
    X_test = X[negindex,:]
    y_train = y[index]
    y_test = y[negindex]
    return(X_train,X_test,y_train,y_test)
    
def data_split(X,y_col=0):
    """Pre-processing for image data. Splits matrix into
    predictors and observation"""
    
    y_train = X[:,y_col]
    X_train = np.delete(X,y_col,axis=1)
    
    return(X_train,y_train)
    
def y_encode(y,y_obs,y_neg=-1):
    """Encodes the positive y-value"""
    
    if y_neg == -1:
            y_train = -1 + 2*(y==y_obs)
    elif y_neg ==0:
        y_train = 1*(y==y_obs)
        
    return(y_train)
            
    
def preprocessing(X,y_col=0,y_obs= None,y_neg = -1):
    """
    Pre-processing for image data. Splits matrix into
    predictors and observation
    
    Inputs:
        X: data matrix containing both predictors and observations
        y_col: column(s) containing oberservations
        y_obs: value of the positive y observation (for 1vsAll)
        y_neg: value of negative encoding either -1 (default) or 0 
    
    """
    y_train = X[:,y_col]
    
    if y_obs is not None:
        
        if y_neg == -1:
            y_train = -1 + 2*(y_train==y_obs)
        elif y_neg ==0:
            y_train = 1*(y_train==y_obs)
    
    X_train = np.delete(X,y_col,axis=1)
    
    return(X_train,y_train)
    
    
    
    
    
    
    