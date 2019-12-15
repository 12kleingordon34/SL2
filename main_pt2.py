# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:57:01 2019

@author: dantr
"""
import numpy as np
from pattern_generators import JALB

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
    

        
        
    