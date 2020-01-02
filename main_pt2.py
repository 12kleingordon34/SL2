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
from knn import OneNN, kNN
import matplotlib.pyplot as plt

# Linear search

def find_least_m(n, algorithm,X_test,y_test,start=0,neg=-1):
    seed = np.random.randint(10000)
    m = start
    error = np.inf
    while error > 0.1:
        m += 1
        X_train, y_train = JALB(n,m,neg=neg,seed=seed)
        algorithm.train(X_train,y_train)
        error = np.mean(algorithm.predict(X_test)!=y_test)
    return(m)
    

def find_avg_least_m(n,algorithm,start=0,num_runs=10,neg=-1):
    #size = int(np.log(1./0.05)/(2*(0.005)**2)) 
    X_test,y_test = JALB(n,6000,neg=neg)
    M = np.zeros(num_runs)
    for i in range(num_runs):
        if type(algorithm)==kNN:
            M[i] = find_least_m_exp(n,algorithm,X_test,y_test,start=start,neg=neg)
        else:
            M[i] = find_least_m(n,algorithm,X_test,y_test,start=start,neg=neg)
        start = max(0,int(M[i]-10))
    return(np.mean(M),np.std(M))
    

def find_trend_m(algorithm,neg=-1,max_n=17):
    start =0
    M_mean = np.zeros(max_n-1)
    M_std = np.zeros(max_n-1)
    for i in range(2,max_n+1):
        M_mean[i-2],M_std[i-2] = find_avg_least_m(i,algorithm,start=start,neg=neg)
        start = max(0,int(M_mean[i-2]-5))
    return(M_mean,M_std)
    
   
# Exponential search

def find_least_m_exp(n, algorithm,X_test,y_test,start=0,neg=-1):
    seed = np.random.randint(10000)
    m = start
    error = np.inf
    
    def exponential_search(m):
        p = int(np.log2(m))
        while error >0.1:
            p+=1
            m = np.power(2,p)
            X_train, y_train = JALB(n,m,neg=neg,seed=seed)
            algorithm.train(X_train,y_train)
            error = np.mean(algorithm.predict(X_test)!=y_test)
        return(m)
        
    def binary_search(m):
        p2 = int(np.log2(m))
        p1 = p2-1
        low = np.power(2,p1)
        high = m
        while low < high-1:
            mid = low+high/2
            X_train, y_train = JALB(n,mid,neg=neg,seed=seed)
            algorithm.train(X_train,y_train)
            error = np.mean(algorithm.predict(X_test)!=y_test)
            if error < 0.1:
                high = mid
            elif error >0.1:
                low = mid
            elif error == 0.1:
                return(mid)
        return(high)
        
    m = exponential_search(m)
    m = binary_search(m)
    return(m)
        
        
        
        
        
def plot_trend(M_mean,M_std,title):
    n = np.arange(2,M_mean.size+2)
    plt.figure()
    plt.errorbar(n,M_mean,yerr=M_std)
    plt.xlabel("n")
    plt.ylabel("m")
    plt.title("{} Sample Complexity".format(title))
    plt.show()
    
    
if __name__=="__main__":
    # there's definitely a better way to do this
#    A = ["Perceptron()","Winnow()","LinearRegression()",]
#    Atitle = ["Perceptron","Winnow", "Least Squares","One nearest-neighbours"]
#    for j, i in enumerate(A):
#        alg = eval(i)
#        neg = -1
#        if i == "Winnow()":
#            neg = 0
#        mean,std = find_trend_m(alg,neg=neg)
#        plot_trend(mean,std,Atitle[j])

    A = kNN()
    mean,std = find_trend_m(alg,neg=neg)
    plot_trend(mean,std,'OneNN')


