# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:07:41 2019

@author: ucabdbt
"""

def loss_grad_softmax_vectorized(W, X, y, reg):
    """ Compute the loss and gradients using softmax with vectorized version"""
    loss = 0 
    grad = np.zeros_like(W)
    dim, num_train = X.shape

    scores = W.dot(X) # [K, N]
    # Shift scores so that the highest value is 0
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    correct_scores_exp = scores_exp[y, range(num_train)] # [N, ]
    scores_exp_sum = np.sum(scores_exp, axis=0) # [N, ]
    loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    scores_exp_normalized = scores_exp / scores_exp_sum
    # deal with the correct class
    scores_exp_normalized[y, range(num_train)] -= 1 # [K, N]
    grad = scores_exp_normalized.dot(X.T)
    grad /= num_train
    grad += reg * W

    return loss, grad