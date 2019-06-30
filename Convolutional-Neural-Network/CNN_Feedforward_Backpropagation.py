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
    
    
x =np.asarray([3,0,1,2,7,4,1,5,8,9,3,1,2,7,2,5,1,3,0,1,3,1,7,8,4,2,1,6,2,8,2,4,5,2,3,9]) #np.asarray([3,1,2,0,4,2,0,5,7,1,2,4,1,8,2,3,1,5,2,9,5,1,6,2,7,3,1,7,2,3,4,1,3,8,8,9])#np.asarray([3,0,1,2,7,4,1,5,8,9,3,1,2,7,2,5,1,3,0,1,3,1,7,8,4,2,1,6,2,8,2,4,5,2,3,9])
f = np.asarray([1,0,-1,1,0,-1,1,0,-1]) #np.asarray([1,1,1,0,0,0,-1,-1,-1])
x = x.reshape(x.shape[0],)
#padding = np.zeros(f.shape[0], ((0)), 'constant', constant_values=(0))
padding = np.pad(f.reshape(3,3), ((0,0), (0,2)), 'constant', constant_values=(0))
#first_col = np.r_[f, padding.reshape(12,)]
pad = np.zeros(x.shape[0]-padding.shape[0]*padding.shape[1], f.dtype)
first_row = np.r_[padding.reshape(x.shape[0]-pad.shape[0],), pad]
pad = np.zeros(((x.reshape(6,6).shape[0] - f.reshape(3,3).shape[0]+1)**2)-1, f.dtype)
first_col = np.r_[f[0], pad]
#F = linalg.toeplitz(first_col, first_row)
F = linalg.circulant(first_row) #, first_row)
np.sum(np.multiply(F,x), axis=1).reshape(4,4)

'''
padding = np.zeros(f.shape[0] - 1, f.dtype)
first_col = np.r_[f, padding]
first_row = np.r_[f[0], padding]
H = linalg.toeplitz(first_col, first_row)
'''


x =np.asarray([3,0,1,2,7,4,1,5,8,9,3,1,2,7,2,5,1,3,0,1,3,1,7,8,4,2,1,6,2,8,2,4,5,2,3,9]).reshape(6,6)
f = np.asarray([1,0,-1,1,0,-1,1,0,-1]).reshape(3,3)
x_row_num, x_col_num = x.shape 
f_row_num, f_col_num = f.shape
output_row_num = x_row_num + f_row_num - 1
output_col_num = x_col_num + f_col_num - 1
print('output dimension:', output_row_num, output_col_num)
f_zero_padded = np.pad(F, ((output_row_num - f_row_num, 0),
                           (0, output_col_num - f_col_num)),
                        'constant', constant_values=0)

from scipy.linalg import toeplitz

# use each row of the zero-padded F to creat a toeplitz matrix. 
#  Number of columns in this matrices are same as numbe of columns of input signal
toeplitz_list = []
for i in range(f_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
    c = f_zero_padded[i, :] # i th row of the F 
    r = np.r_[c[0], np.zeros(x_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
    toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
    toeplitz_list.append(toeplitz_m)
    print('F '+ str(i)+'\n', toeplitz_m)

c = range(1, f_zero_padded.shape[0]+1)
r = np.r_[c[0], np.zeros(x_row_num-1, dtype=int)]
doubly_indices = toeplitz(c, r)
toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
h = toeplitz_shape[0]*doubly_indices.shape[0]
w = toeplitz_shape[1]*doubly_indices.shape[1]
doubly_blocked_shape = [h, w]
doubly_blocked = np.zeros(doubly_blocked_shape)

# tile toeplitz matrices for each row in the doubly blocked matrix
b_h, b_w = toeplitz_shape # hight and withs of each block
for i in range(doubly_indices.shape[0]):
    for j in range(doubly_indices.shape[1]):
        start_i = i * b_h
        start_j = j * b_w
        end_i = start_i + b_h
        end_j = start_j + b_w
        doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

print('doubly_blocked: ', doubly_blocked)




