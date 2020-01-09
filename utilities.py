# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:02:06 2019

@author: dantr
"""
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratified_k_fold(P, X, y, percentage=0.2, epochs=1, seed=0, conf=True):
    """
    Runs a stratified k-fold cross validation on
    perceptron P using dataset (X, y). Split can
    be controlled through a choice of seed
    """
    classification_accuracy = []
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=percentage, random_state=seed, stratify=y
    )
    for e in range(epochs):
        y_pred = perceptron_learning(
            P, X_train, X_test, y_train
        )
    train_error = (P.predict(X_train) == y_train).mean()
    test_error = (y_pred == y_test).mean()
    print("Train error: {} Test error: {}".format(train_error, test_error))
    if conf:
        y_confusion = np.concatenate(
            (y_pred[None,:].T, y_test[None,:].T),
            axis=1
        )
        # Select only incorrect predictions
        y_confusion = y_confusion[
            (y_confusion[:,0] != y_confusion[:,1])
        ]
        return train_error, test_error, y_confusion
    else:
        return train_error, test_error


def multip_strat_kfold(P_list, X, y, k, epochs=1, seed=0):
    """
    Runs a stratified k-fold cross validation on
    a list of perceptrons list_P using dataset (X, y). Split can
    be controlled through a choice of seed

    THIS IS A TEST, MAY NOT BE NECESSARY. Note that this is not
    vectorised. May need to work on this later.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accuracy_error = []
    k_val= 1
    for train_index, test_index in skf.split(X, y):
        epoch_error = []
        for e in range(epochs):
            predictions = np.zeros((len(test_index), len(P_list)))
            for i, P in enumerate(P_list):
                acc = []
                X_train, X_test = X[train_index], X[test_index]
                print("TIME: {} X-Val: {} Epoch: {} Training Perceptron {}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), k_val, e+1, i
                ))
                y_train = y_encode(y[train_index], i)
                y_test = y_encode(y[test_index], i)
                y_prob = perceptron_learning(
                    P, X_train, X_test, y_train, y_test, num_epoch=1
                )
                predictions[:, i] = y_prob
            predictions = np.argmax(predictions, axis=1)
            error = (predictions == y[test_index]).mean()
            epoch_error.append(error)
        k_val += 1
        accuracy_error.append(epoch_error)
    return accuracy_error


def vectorised_p_strat_kfold(P, X, y, k, epochs=1, seed=0):
    """
    Runs a stratified k-fold cross validation on
    a list of perceptrons list_P using dataset (X, y). Split can
    be controlled through a choice of seed

    THIS IS A TEST, MAY NOT BE NECESSARY. Note that this is not
    vectorised. May need to work on this later.
    """
    print("Seed: {}".format(seed))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accuracy_error = []
    k_val= 1
    for train_index, test_index in skf.split(X, y):
        epoch_error = []
        for e in range(epochs):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("TIME: {} X-Val: {} Epoch: {}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), k_val, e+1
            ))
            y_pred = perceptron_learning(
                P, X_train, X_test, y_train 
            )
            error = (y_pred == y_test).mean()
            train_error = (P.predict(X_train) == y_train).mean()
            epoch_error.append(error)
            print("Test error: {} Train error: {}".format(error, train_error))
        k_val += 1
        accuracy_error.append(epoch_error)
    return accuracy_error


def perceptron_learning(P, X_train, X_test, y_train):
    """
    Trains perceptron P with (X_train, y_train) and evaluates
    on (X_test, y_test). Returns the prediction accuracy and
    the indices of missclassified elements.
    """
    P.train(X_train, y_train)
    y_pred = P.predict(X_test)

    return y_pred

    
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
    
    
    
    
    
    
    
