# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:02:06 2019

@author: dantr
"""
import numpy as np
def train_test_split(X,y,percentage=1/3):
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
    perm = np.random.permutation(np.arange(l))
    # split data proportionally
    index = perm[:int((1-percentage)*l)]
    negindex = perm[int((1-percentage)*l):]
    X_train = X[index,:]
    X_test = X[negindex,:]
    y_train = y[index]
    y_test = y[negindex]
    return(X_train,X_test,y_train,y_test)
    
    
def preprocessing(X,y_col=0):
    """
    Pre-processing for image data. Splits matrix into
    predictors and observation
    """
    y_train = X[:,y_col]
    X_train = np.delete(X,y_col,axis=1)
    
    return(X_train,y_train)
    
    