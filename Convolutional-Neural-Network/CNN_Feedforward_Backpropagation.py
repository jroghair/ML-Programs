# -*- coding: utf-8 -*-
"""
Created on Thu Jun 03 19:00:20 2019

@author: jrogh

This is a simple program to help me understand exactly how forward/back propagation works for convolutional neural networks. 
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import correlate2d


#Adds padding to an image matrix
def get_padding(X, n):
    X_padded = np.pad(X,((0,0),(n,n),(n,n),(0,0)),'constant',constant_values=(0,0))
    return X_padded

#Performs a single convolution step to generate a scalar value as a result of muliplying the filter and the image matrix
def perform_convolution(one_slice, weights, bias):
        p = np.multiply(one_slice, weights)
        val = np.sum(p)
        val+=float(bias)
        return val

def convolution_feedforward_layer(X_prev, weights, bias, hparam): #stride, padding):
    #previous layer shapes 
    (m, n_row_prev, n_col_prev, n_depth_prev) = X_prev.shape
    #Filter dimensions
    (f, f, n_depth_prev, n_depth) = weights.shape
    stride, padding = hparam["stride"], hparam["padding"]
    #Output dimensions for the current layer, dimensions output will be #floor(n-f+2*padding/stride) + 1
    n_row= int((n_row_prev - f + 2*padding)/stride)+1
    n_col = int((n_col_prev - f + 2*padding)/stride)+1
    #Created output dimenions for this layer
    out = np.zeros((m, n_row, n_col, n_depth))
    #Add padding to the previous layer so output dimensions remain the same
    X_prev_padded = get_padding(X_prev, padding)
    
    for i in range(m):
        for row in range(n_row):
            for col in range(n_col):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)
                    x_slice_prev = X_prev_padded[i, rs:re, cs:ce, :]
                    out[i, row, col, d] = perform_convolution(x_slice_prev, weights[:,:,:,d], bias[:,:,:,d])
    delta = (X_prev, weights, bias, hparam) 
    return out, delta 

def pooling_forward_layer(X_prev, hparam, method="max"):
    (m, n_row_prev, n_col_prev, n_depth_prev) = X_prev.shape
    
    stride, f = hparam["stride"], hparam["f"]
    
    n_row= int(1+(n_row_prev - f )/stride)
    n_col = int(1+(n_col_prev - f)/stride)
    n_depth = n_depth_prev
    
    out = np.zeros((m, n_row, n_col, n_depth))
    for i in range(m):
        for row in range(n_row):
            for col in range(n_col):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)
                    if method == "max":
                            out[i, row, col, d] = np.max(X_prev[i, rs:re, cs:ce, d])
                    else:
                        out[i, row, col, d] = np.mean(X_prev[i, rs:re, cs:ce, d])
    delta = (X_prev, hparam) 
    return out, delta           
                    
                    
def convolution_backpropogation_layer(dJ, delta):
    (X_prev, weights, bias, hparam) = delta
    (m, n_row_prev, n_col_prev, n_depth_prev) = X_prev.shape
    (f, f, n_depth_prev, n_depth) = weights.shape
    
    #get dimension of gradient of the cost function 
    (m, n_row, n_col, n_depth) = dJ.shape 
    
    stride, padding = hparam["stride"], hparam["padding"]
    
    #Dimensions of Output layer backwards
    dX_prev = np.zeros((m, n_row_prev, n_col_prev, n_depth_prev))
    dWeights = np.zeros((f, f, n_depth_prev, n_depth))
    dbias = np.zeros((f,f, n_depth_prev, n_depth))
    
    #Add padding
    X_prev_padded = get_padding(X_prev, padding)
    dX_prev_padded = get_padding(dX_prev, padding)
    
    #perform backpropogation
    for i in range(m):
        for row in range(n_row):
            for col in range(n_col):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)    
                    x_slice = X_prev_padded[i, rs:re, cs:ce, :]
                    dX_prev_padded[i, rs:re, cs:ce,:] += weights[:, :, :, d]*dJ[i, row, col, d]
                    dWeights[:, :, :, d] += x_slice*dJ[i, row, col, d]
                    dbias[:, :, :, d] += dJ[i, row, col, d]
        dX_prev[i,:,:,:] = dX_prev_padded[i, padding:-padding, padding:-padding, :]
    return dX_prev, dWeights, dbias


def pooling_backpropogation_layer(dX, delta, method="max"):                   
    (X_prev, hparam) = delta
    
    stride, f = hparam["stride"] , hparam["f"]
    
    m, n_row_prev, n_col_prev, n_depth_prev = X_prev.shape
    m, n_row, n_col, n_depth = dX.shape
    
    #output dimensions
    dX_prev = np.zeros_like(X_prev)
    
    for i in range(m):
        for row in range(n_row):
            for col in range(n_col):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)    
                    if method == "max":
                        x_prev_slice = X_prev[i, rs:re, cs:ce, d]
                        largest = (x_prev_slice == np.max(x_prev_slice))
                        dX_prev[i, rs:re, cs:ce, d] += largest*dX[i, rs, cs, d]
                    elif method == "average":
                        dx = dX[i,rs,cs,d]
                        avg = np.ones((f,f))*(dx/(f*f))
                        dX_prev[i,  rs:re, cs:ce, d] += avg
    return dX_prev

    
          
    
                    