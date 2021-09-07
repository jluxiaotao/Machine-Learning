# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:33:41 2021

@author: PengTao
"""
import numpy as np
class RLS:
    def __init__(self,lamda,M):
        """
        Parameters
        ----------
        lamda : 0<lamda<=1
            forgetting factor.
        M : points number.
            length of data used for minimize mean square of error.
        Returns
        -------
        None.

        """
        self.lamda=lamda
        self.M=M
        self.delta=1e6
        self.P=np.mat(self.delta*np.eye((self.M)))
        self.w=np.mat(np.zeros((self.M))).T
    def fit(self,x,d):
        """
        Parameters
        ----------
        x : input array.
            DESCRIPTION.
        d : desire output.
            DESCRIPTION.
        Returns
        -------
        None.

        """
        N=len(x)
        for i in range(self.M,N):
            xi=np.mat(x[i-self.M:i][::-1])
            di=np.mat(d[i])
            e=di-xi*self.w
            k=self.P*xi.T/(self.lamda+xi*self.P*xi.T)
            self.P=(self.P-k*xi*self.P)/self.lamda
            self.w=self.w+k*e  
    def rls_filter(self,u):
        N=len(u)
        y=np.zeros(N)
        for i in range(N):
            if i<self.M:
                y[i]=u[i]
            else:
                y[i]=u[i-self.M:i][::-1]*self.w
        return y
    def rls_derivate(self,u,t):
        N=len(u)
        dy=np.zeros(N)
        for i in range(N):
            if i<self.M+1:
                dy[i]=u[i]
            else:
                delta_ui=(u[i-self.M:i]-u[i-self.M-1:i-1])[::-1]
                delta_ti=(t[i-self.M:i]-t[i-self.M-1:i-1])[::-1]
                dy[i]=delta_ui/delta_ti*self.w
        return dy
        
from matplotlib import pyplot as plt
 
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
 
t = np.arange(0, 5000) * 0.01
x = np.sin(t)
n = wgn(x, 30)
xn = x+n
rls=RLS(0.98,3)
rls.fit(xn,x)

n = wgn(x, 40)
xn = x+n
plt.plot(t,xn)
# # plt.show()
plt.plot(t,dx)
# # plt.show()


y=rls.rls_filter(xn)
y=rls.rls_filter(y)
y=rls.rls_filter(y)
dy=rls.rls_derivate(y,t)
plt.plot(t,y)
# plt.show()
plt.plot(t,dy)
plt.show()
