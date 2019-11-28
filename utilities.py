# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:02:06 2019

@author: dantr
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratified_basic_validation(P, X, y, percentage=1/5, n=1, seed=0):
    """ Splits a dataset into training and test sets
        with a random permutation
    
        Inputs:
            X: data matrix
            y: values
            percentage: proportion of data in test set
        Outputs:
            X_train,X_test,y_train,y_test: split data
    """
    classification_accuracy = []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=percentage, random_state=seed+i
        ) acc, failed_pred = perceptron_learning( P, X_train, X_test, y_train, y_test
        )
        classification_accuracy.append(acc)
    
    return classification_accuracy


def stratified_k_fold(P, X, y, k, n=1, seed=0):
    """
    Runs a stratified k-fold cross validation on
    perceptron P using dataset (X, y). Split can
    be controlled through a choice of seed
    """
    classification_accuracy = []
    for i in range(n):
        skf = StratifiedKFold(n_splits=k, random_state=seed)
        k_fold_acc = []
        for train_i, test_i in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            acc, failed_pred = perceptron_learning(
                P, X_train, X_test, y_train, y_test
            )
            k_fold_acc.append(acc)
        classification_accuracy.append(k_fold_acc)
    return classification_accuracy


def multiperceptron_stratified_k_fold(P_list, X, y, k, n=1, seed=0):
    """
    Runs a stratified k-fold cross validation on
    a list of perceptrons list_P using dataset (X, y). Split can
    be controlled through a choice of seed

    THIS IS A TEST, MAY NOT BE NECESSARY. Note that this is not
    vectorised. May need to work on this later.
    """
    class_acc = {P.k_params: [] for P in P_list}
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    for train_index, test_index in skf.split(X, y):
        print(type(P_list))
        predictions = np.zeros((len(y), len(P_list)))
        for i, P in enumerate(P_list):
            acc = []
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_encode(y[train_index], i), y_encode(y[test_index], i)
            y_prob = perceptron_learning(
                P, X_train, X_test, y_train, y_test, num_epoch=1
            )
            predictions[:, i] = y_prob
    predictions = np.argmax(predictions, axis=1)
    return predictions, class_acc


def perceptron_learning(P, X_train, X_test, y_train, y_test, num_epoch):
    """
    Trains perceptron P with (X_train, y_train) and evaluates
    on (X_test, y_test). Returns the prediction accuracy and
    the indices of missclassified elements.
    """
    P.train(X_train, y_train, num_epoch)
    y_prob = P.predict_proba(X_test)

    return y_prob

    
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
    
    
    
    
    
    
    
