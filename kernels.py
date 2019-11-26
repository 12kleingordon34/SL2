# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:09:58 2019

@author: dantr
"""
import numpy as np
def gaussian_kernel_row(data,anchor,sigma):
    """Gaussian kernel for an array of points with parameter sigma
    Returns one row of the kernel matrix.
    """
    return np.exp(-sigma * np.sum((data-anchor)*(data-anchor),axis=1))



def radial_basis_kernel(s,t,sigma):
    """ Builds a Gaussian Kernel matrix
        Inputs:
            s: data matrix
            t: data matrix
            sigma: bandwidth
        Outputs:
            K: Kernel matrix
    """
    l1,n1 = s.shape
    l2,n2 = t.shape
    assert n1==n2
    
    K = np.zeros((l1,l2))
    for i in range(l1):
        K[i,:] = gaussian_kernel_row(X,X[i,:],sigma)
    return(K)
    
def polynomial_kernel(X, Y, d):
    """
    Calculates a d-order polynomial kernel for
    data points x_i, x_j
    """
    #n1 = X.shape[0]
    #n2 = np.atleast_2d(Y).T.shape[0]
    product = np.dot(X, np.atleast_2d(Y).T)
    return np.power(product, d)