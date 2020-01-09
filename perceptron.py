# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:18:03 2019

@author: ucabdbt
"""
import numpy as np
from scipy import stats


class Perceptron(object):
    """
    Single class non-kernel perceptron
    """
    def __init__(self):
        self.w = np.array([]) # weights
        self.num_classes = 1 # number of classes
        self.M = 0 # number of mistakes
        self.data_hash = None

    def train(self,X,y):
        """Train function. Allows for multiple reruns on same data (epochs)"""
        m,d = X.shape
        self.num_classes = np.atleast_2d(y).shape[0]
        # check if data is new else do not reset weights
        training_hash = hash(tuple(y))
        if self.data_hash != training_hash:
            self.data_hash = training_hash
            self.w = np.zeros(d)
        self._training_run(X, y, m) # train
        
    def _training_run(self, X, y, m):
        """Performs a training run"""
        for t in range(m):
            yhat = np.sign(np.dot(X[t,:],self.w))
            if yhat*y[t] <= 0:
                self.w += y[t]*X[t,:]
                self.M += 1
                

    def predict_proba(self,X):
        """Returns margin on new data X"""
        if self.w.shape[0]==self.w.size:
           return(np.dot(X,self.w).flatten())
        return(np.dot(X,self.w))
        

    def predict(self,X):
        """Predicts class for new data X"""
        return(np.sign(self.predict_proba(X)))


class KernelPerceptron(object):
    """
    Single class kernelised perceptron.
    """
    def __init__(self,kernel,k_params):
        self.w = np.array([])
        self.num_classes = 1
        self.M = 0
        self.train_set = 0
        self.k_params = k_params
        self.kernel = kernel
        self.data_hash = None
        

    def build_gram(self,X):
        return(self.kernel(X,X,self.k_params))

    def train(self, X, y):
        self.train_set = X
        m,d = X.shape
        self.num_classes = np.atleast_2d(y).shape[0]
        self.M = np.zeros(self.num_classes)
        gram = self.build_gram(X)
        training_hash = hash(tuple(y))
        if self.data_hash != training_hash:
            self.data_hash = training_hash
            self.w = np.zeros(m)

        for i in range(m):
            if np.sign(np.dot(self.w, gram[:,i])) != y[i]:
                self.w[i] += y[i]
                self.M+=1

    def predict_proba(self,x):
        k = self.kernel(self.train_set,x,self.k_params)
        return(np.dot(self.w,k))
        

    def predict(self,x):
        return(np.sign(self.predict_proba(x)))


class VectorisedKernelPerceptron(object):
    """
    Multiclass 1-vs-all Kernelised Perceptron
    """
    def __init__(self,kernel,k_params):
        self.W = np.array([])
        self.num_classes = 1
        self.M = 0
        self.train_set = 0
        self.k_params = k_params
        self.kernel = kernel
        self.data_hash = None
        

    def build_gram(self,X):
        return(self.kernel(X,X,self.k_params))

    def train(self, X, y):
        self.train_set = X
        m,d = X.shape

        # Encode responses
        y = y.astype(int)
        n_vals = np.max(y) + 1
        Y = 2*np.eye(n_vals)[y] - 1

        self.num_classes = np.atleast_2d(y).shape[0]
        self.M = np.zeros(self.num_classes)
        gram = self.build_gram(X)
        # Reset W matrix if the training data is reset. This is not a
        # problem if we train multiple epochs over the same training dataset.
        # This was done to simplify the calculation of the kernel gram matrix
        training_hash = hash(tuple(y))
        if self.data_hash != training_hash:
            self.data_hash = training_hash
            self.W = np.zeros((m, n_vals))

        for i in range(m):
            # Calculate w.x for all 10 classifiers
            beta = np.dot(gram[i,:], self.W) 
            gamma = (np.sign(beta) != Y[i,:])

            # Apply update if prediction was incorrect
            self.W[i,:] += np.multiply(Y[i,:], gamma)

    def predict_proba(self,x):
        k = self.kernel(self.train_set,x,self.k_params)
        return np.dot(k.T, self.W)
        
    def predict(self,x):
        y_prob = self.predict_proba(x)
        # Select the classifier with the largest margin
        return np.argmax(y_prob, axis=1)


class onevsonePerceptron(object):
    """
    Multiclass 1-vs-1 Kernelised Perceptron
    """
    def __init__(self,kernel,k_params):
        self.W = np.array([])
        self.num_classes = 1
        self.M = 0
        self.train_set = 0
        self.k_params = k_params
        self.kernel = kernel
        self.data_hash = None
        self.p_index = np.genfromtxt('onevsonepairs.csv', delimiter=',')
        

    def build_gram(self,X):
        return(self.kernel(X,X,self.k_params))

    def train(self, X, y):
        self.train_set = X
        m,d = X.shape
        n_vals = self.p_index.shape[0]
        self.num_classes = np.atleast_2d(y).shape[0]
        self.M = np.zeros(self.num_classes)
        gram = self.build_gram(X)

        # Reset W matrix if the training data is reset. This is not a
        # problem if we train multiple epochs over the same training dataset.
        # This was done to simplify the calculation of the kernel gram matrix
        training_hash = hash(tuple(y))
        if self.data_hash != training_hash:
            self.data_hash = training_hash
            self.W = np.zeros((m, n_vals))

        for i in range(m):
            y_m = y[i]
            Y = (self.p_index[:, 0] == y_m)
            Y = Y - (self.p_index[:, 1] == y_m).astype(int)
            beta = np.dot(gram[i,:], self.W) 
            gamma = (np.sign(beta) != Y)

            # Apply update if prediction was incorrect
            self.W[i,:] += np.multiply(Y, gamma)

    def predict_proba(self,x):
        k = self.kernel(self.train_set,x,self.k_params)
        return np.dot(k.T, self.W)

    def predict(self,x):
        # Run all K(K-1)/2 classifiers and take majority vote
        y_prob = self.predict_proba(x)
        y_prob = np.sign(y_prob)
        predictions = np.zeros(y_prob.shape)
        # Fill predictions with the classes each classifier predicts
        for row in range(y_prob.shape[0]):
            predictions[row,:][y_prob[row,:]==1] = self.p_index[y_prob[row,:]==1,0]
            predictions[row,:][y_prob[row,:]==-1] = self.p_index[y_prob[row,:]==-1,1]
        return stats.mode(predictions, axis=1).mode.T
