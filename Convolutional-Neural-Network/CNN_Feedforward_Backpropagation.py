# -*- coding: utf-8 -*-
"""
Created on Thu Jun 03 19:00:20 2019

@author: jrogh
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import linalg



#Adds padding to an image matrix
def padding(X, n):
    X_padded = np.pad(X,((0,0),(n,n),(n,n),(0,0)),'constant',constant_values=(0,0))
    return X_padded

#Performs a single convolution step to generate a scalar value as a result of muliplying the filter and the image matrix
def Convolution_step(one_slice, weights, bias):
        p = np.multiply(one_slice, weights)
        val = np.sum(p)
        val+=float(bias)
        return val
    


def convolution_feedforward(X_prev, weights, bias, stride, padding):
    #previous layer shapes 
    (m, n_H_prev, n_W_prev, n_depth_prev) = X_prev.shape
    #Filter dimensions
    (f, f, n_depth_prev, n_depth) = weights.shape
    
    #Output dimensions for the current layer, dimensions output will be #floor(n-f+2*padding/stride) + 1
    n_H = int((n_H_prev - f + 2*padding)/stride)+1
    n_W = int((n_W_prev - f + 2*padding)/stride)+1
    
    #Created output dimenions for the 
    out = np.zeros((m, n_H, n_W, n_depth))
    
    #Add padding to the previous layer so output dimensions remain the same
    X_prev_padded = padding(X_prev, padding)
    
    for i in range(m):
        x_prev_padded = X_prev_padded[i] 
        c_st, c_end, r_st, r_end = (stride*)
    
    
x = np.asarray([3,0,1,2,7,4,1,5,8,9,3,1,2,7,2,5,1,3,0,1,3,1,7,8,4,2,1,6,2,8,2,4,5,2,3,9])
f = np.asarray([1,0,-1,1,0,-1,1,0,-1])
x = x.reshape(x.shape[0],)
#padding = np.zeros(f.shape[0], ((0)), 'constant', constant_values=(0))
padding = np.pad(f.reshape(3,3), ((0,0), (0,1)), 'constant', constant_values=(0))
#first_col = np.r_[f, padding.reshape(12,)]
pad = np.zeros(24, f.dtype)
first_col = np.r_[padding.reshape(12,), pad]
pad = np.zeros(f.reshape(3,3).shape[0]-1, f.dtype)
first_row = np.r_[f[0], pad]
F = linalg.toeplitz(first_col, first_row)


'''
padding = np.zeros(f.shape[0] - 1, f.dtype)
first_col = np.r_[f, padding]
first_row = np.r_[f[0], padding]
H = linalg.toeplitz(first_col, first_row)
'''