# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:28:47 2021

@author: PengTao
"""
import numpy as np
a=np.array([[2,3],[2,3]])
b=np.array([[1,2,3,4],[2,3,4,5]])
d=np.array([1,1,1,1])
c=np.dot(a,b)+d
print(c)
e=np.exp(c-np.max(c,axis=1,keepdims=True))
print(e)
e/=np.sum(e,axis=1,keepdims=True)
print(e)
t=np.array([0,1])
print(e[np.arange(2), t])