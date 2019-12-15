# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:57:01 2019

@author: dantr
"""
import numpy as np
from pattern_generators import JALB
from least_squares import LinearRegression
from winnow import Winnow
from perceptron import Perceptron
from knn import OneNN

def find_least_m(n, algorithm,X_test,y_test):
    
    seed = np.random.randint(10000)
    
    m = 0
    
    error = np.inf
    
    while error > 0.1:
        m += 1
        
        X_train, y_train = JALB(n,m,seed)
        
        algorithm.train(X_train,y_train)
        
        error = np.mean(algorithm.predict(X_test)==y_test)
        
    return(m)
    
def find_avg_least_m(n,algorithm,num_runs=50):
    
    X_test,y_test = JALB(n,min(pow(2,n),1000))
    
    M = np.zeros(num_runs)
    
    for i in range(num_runs):
        
        M[i] = find_least_m(n,algorithm,X_test,y_test)
        
    return(np.mean(M),np.std(M))
    


    
    
        
        
    