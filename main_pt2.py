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
import matplotlib.pyplot as plt

def find_least_m(n, algorithm,X_test,y_test,start=0):
    
    seed = np.random.randint(10000)
    
    m = start
    
    error = np.inf
    
    while error > 0.1:
        m += 1
        
        X_train, y_train = JALB(n,m,seed)
        
        algorithm.train(X_train,y_train)
        
        error = np.mean(algorithm.predict(X_test)!=y_test)
        
    return(m)
    
def find_avg_least_m(n,algorithm,start=0,num_runs=50):
    
    X_test,y_test = JALB(n,min(pow(2,n),5000))
    
    M = np.zeros(num_runs)
    
    for i in range(num_runs):
        
        M[i] = find_least_m(n,algorithm,X_test,y_test,start)
        start = max(0,int(M[i]-10))
        
    return(np.mean(M),np.std(M))
    

def find_trend_m(algorithm):
    start =0
    M_mean = np.zeros(99)
    M_std = np.zeros(99)
    
    for i in range(2,101):
        M_mean[i-2],M_std[i-2] = find_avg_least_m(i,algorithm,start)
        start = max(0,int(M_mean[i-2]-5))
    
    return(M_mean,M_std)
    
    
def plot_trend(M_mean,M_std):
    
    n = np.arange(2,M_mean.size+2)
    plt.figure()
    plt.errorbar(n,M_mean,yerr=M_std)
    plt.xlabel("n")
    plt.ylabel("m")
    plt.show()
    
    
if __name__=="__main__":
    alg = LinearRegression()
    mean,std = find_trend_m(alg)
    plot_trend(mean,std)
    
    
    
        
    
    
    
        
        
    