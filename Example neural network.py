# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:28:47 2021

@author: PengTao
"""
import numpy as np
def affine_forward(x,w,b):
    """
    Return the result of H=xw+b and cache of w,b.
    Parameters
    ----------
    x : numpy array
        An array containing a number of inputs;
        For example:input is d_1x...xd_k matrix,number of inputs is n,X is a nxd_1x...xd_k matrix.
    w : gain matrix
        A matrix with dimension d_1*...*d_kxp, p is number of neurons.
    b : offset matrix
        A matrix with dimension nxp.

    Returns
    -------
    result : numpy array
        Return the result of H=xw+b and cache of w,b.
    """
    out=None
    N=x.shape[0]
    x_row=x.reshape(N,-1)
    out=np.dot(x_row,w)+b
    cache=(x,w,b)
    return out,cache
def affine_backward(dout,cache):
    """
    Return the backward propagation result of H=wx+b.
    dw=dH*∂H/∂w, dx=dH*∂H/∂x, db=dH*∂H/∂b
    Parameters
    ----------
    dout : numpy array
        Output of Hidden layer, input of Output layer.
        An array with dimension nxN, N is the number of groups.
    cache : numpy array
        Cache of x, w, b, dimension of x, w, b is not equal.
        
    Returns
    -------
    result : numpy array
         Return the backward propagation result of H=wx+b, dx, dw, db.
    """
    x,w,b=cache
    dx,dw,db=None,None,None
    dx=np.dot(dout,w.T)
    dx=np.reshape(dx,x.shape)
    x_row=x.reshape(x.shape[0],-1)
    dw=np.dot(x_row.T,dout)
    db=np.sum(dout,axis=0,keepdims=True)
    return dx,dw,db
"""
Initialize input and parameters
"""
x=np.array([[2,1],
            [-1,1],
            [-1,-1],
            [1,-1]])
tag=np.array([1,2,3,4])

input_dim=x.shape[1]
num_groups=tag.shape[0]
hidden_dim=50
reg=0.001
epsilon=0.001
num_train=10000

#np.random.seed(1)
w1=np.random.randn(input_dim,hidden_dim)
w2=np.random.randn(hidden_dim,num_groups)
b1=np.zeros((1,hidden_dim))
b2=np.zeros((1,num_groups))
"""
Train
"""
for i in range(num_train):
    H,f1_cache=affine_forward(x,w1,b1)              # Forward propagation 1st layer
    H=np.maximum(0,H)                               # Activate by iReLU formula
    H_cache=H
    Y,f2_cache=affine_forward(H,w2,b2)              # Forward propagation 2nd layer
    probs=np.exp(Y-np.max(Y,axis=1,keepdims=True))
    probs/=np.sum(probs,axis=1,keepdims=True)       # Softmax
    print(probs)
    loss=0
    for j in range(x.shape[0]):
        loss+=-(np.log(probs[j,tag[j]-1]))/x.shape[0]
    print(loss)
    dout=np.copy(probs)
    for j in range(x.shape[0]):
        dout[j,tag[j]-1]-=1
    dout/=x.shape[0]
    dh2,dw2,db2=affine_backward(dout,f2_cache)
    dh2[H_cache<=0]=0
    dh1,dw1,db1=affine_backward(dh2,f1_cache)
    
    dw2 += reg * w2
    dw1 += reg * w1
    w2 += -epsilon * dw2
    b2 += -epsilon * db2
    w1 += -epsilon * dw1
    b1 += -epsilon * db1
"""
Test
"""
test=np.array([[1,100],
               [-278,2],
               [-2,-288],
               [2,-1112]]) 
H,f1_cache=affine_forward(test,w1,b1)
H=np.maximum(0,H)
Y,f2_cache=affine_forward(H,w2,b2)
probs=np.exp(Y-np.max(Y,axis=1,keepdims=True))
probs/=np.sum(probs,axis=1,keepdims=True)
print(probs)
for i in range(test.shape[0]):
    print(test[i,:],"at",np.argmax(probs[i,:])+1)
        