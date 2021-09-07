# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:08:00 2021

@author: PengTao
"""
import numpy as np
import random
class SVR:
    def __init__(self,C,tolerance,maxloop):
        """
        Parameters
        ----------
        charaMat : matrix of nXm.
            n is the number of train samples.
            m is the dimension of character vector.
        label : 1Xn vector.
            If label is a nX1 vector,.transpose need to be moved.
        C : TYPE
            DESCRIPTION.
        tolerance : TYPE
            DESCRIPTION.
        maxloop : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.cM=None
        self.lb=None
        self.C=C
        self.tl=tolerance
        self.maxloop=maxloop
        self.b=0
        self.ei_alpha=0
        self.ej_beta=0
        self.w=None
    def Ei(self,i):
        ei=(self.betas-self.alphas).T*self.cM*self.cM[i].T+self.b-self.lb[i]
        return ei
    def ita(self,i,j):
        Ita=self.cM[i]*self.cM[i].T-2*self.cM[i]*self.cM[j].T+self.cM[j]*self.cM[j].T
        return Ita
    def clip(self,i,j,beta_new_unclip):
        L=max(0,self.betas[j]-self.alphas[i])
        H=min(self.C,self.C+self.betas[j]-self.alphas[i])
        if beta_new_unclip<L:
            beta_new=L
        elif beta_new_unclip>H:
            beta_new=H
        else:
            beta_new=beta_new_unclip
        return beta_new
    # def clip_aa(self,i,j,alpha_new_unclip):
    #     L=max(0,self.alphas[j]+self.alphas[i]-self.C)
    #     H=min(self.C,self.alphas[j]+self.alphas[i])
    #     if alpha_new_unclip<L:
    #         alpha_new=L
    #     elif alpha_new_unclip>H:
    #         alpha_new=H
    #     else:
    #         alpha_new=alpha_new_unclip
    #     return alpha_new
    def updateb(self,i,j,alpha_cache,beta_cache):
        if self.betas[j]>0 and self.betas[j]<self.C:
            self.b+=(self.tl-self.ej_beta+(self.alphas[i]-alpha_cache)*self.cM[i]*self.cM[j].T+ (beta_cache-self.betas[j])*self.cM[j]*self.cM[j].T)
        elif self.alphas[i]>0 and self.alphas[i]<self.C:
            self.b+=(self.tl-self.ei_alpha+(self.alphas[i]-alpha_cache)*self.cM[i]*self.cM[i].T+ (beta_cache-self.betas[j])*self.cM[i]*self.cM[j].T)
        else:
            b_a=(self.tl-self.ei_alpha+(self.alphas[i]-alpha_cache)*self.cM[i]*self.cM[i].T+ (beta_cache-self.betas[j])*self.cM[i]*self.cM[j].T)
            b_b=(self.tl-self.ej_beta+(self.alphas[i]-alpha_cache)*self.cM[i]*self.cM[j].T+ (beta_cache-self.betas[j])*self.cM[j]*self.cM[j].T)
            self.b+=(b_a+b_b)/2 
    # def updateb_aa(self,i,j,alphai_cache,alphaj_cache):
    #     if self.alphas[j]>0 and self.alphas[j]<self.C:
    #         self.b+=(self.tl-self.ej_beta+(self.alphas[i]-alphai_cache)*self.cM[i]*self.cM[j].T+ (self.alphas[j]-alphaj_cache)*self.cM[j]*self.cM[j].T)
    #     elif self.alphas[i]>0 and self.alphas[i]<self.C:
    #         self.b+=(self.tl-self.ei_alpha+(self.alphas[i]-alphai_cache)*self.cM[i]*self.cM[i].T+ (self.alphas[j]-alphaj_cache)*self.cM[i]*self.cM[j].T)
    #     else:
    #         b_a=(self.tl-self.ei_alpha+(self.alphas[i]-alphai_cache)*self.cM[i]*self.cM[i].T+ (self.alphas[j]-alphaj_cache)*self.cM[i]*self.cM[j].T)
    #         b_b=(self.tl-self.ej_beta+(self.alphas[i]-alphai_cache)*self.cM[i]*self.cM[j].T+ (self.alphas[j]-alphaj_cache)*self.cM[j]*self.cM[j].T)
    #         self.b+=(b_a+b_b)/2
    def selectj(self,i,loop):
        maxe=0
        optj=None
        jcache=[i]
        self.ei_alpha=self.Ei(i)
        for j in range(self.n):
            if j==i:
                continue
            else:
                kkt=self.lb[j]-(self.betas-self.alphas).T*self.cM*self.cM[j].T-self.b-self.tl
                if kkt>0 and self.betas[j]<self.C and self.betas[j]>=0:
                    if loop==0:
                        e=(self.Ei(j)-self.ei_alpha)
                    else:  
                        e=abs(self.Ei(j)-self.ei_alpha)
                    jcache.append(j)
                    if e>maxe:
                        maxe=e
                        optj=j
        if optj==None:
            for j in range(self.n):
                if j not in jcache:
                    e=abs(self.Ei(j)-self.ei_alpha)
                    if e>maxe:
                        maxe=e
                        optj=j
        if optj!=None:
            self.ej_beta=self.Ei(optj)
        return optj
    def fit(self,charaMat,label):
        self.cM=np.mat(charaMat)
        self.lb=np.mat(label).transpose()
        self.n=self.cM.shape[0]
        self.alphas=np.mat(np.zeros((self.n,1)))
        self.betas=np.mat(np.zeros((self.n,1)))
        loop=0
        while loop<self.maxloop:    
            for i in range(self.n):
                if loop==0:
                    optj=self.selectj(i,loop)
                elif self.alphas[i]<self.C and self.alphas[i]>0:
                    optj=self.selectj(i,loop)
                else:
                    continue
                if optj==None:
                    continue
                beta_new_unclip=self.betas[optj]+(self.ei_alpha-self.ej_beta-2*self.tl)/self.ita(i,optj)
                beta_new=self.clip(i,optj,beta_new_unclip)
                alpha_cache=self.alphas[i]
                beta_cache=self.betas[optj]
                self.alphas[i]=self.alphas[i]+beta_new-self.betas[optj]
                self.betas[optj]=beta_new
                self.updateb(i,optj,alpha_cache,beta_cache)
            # for i in range(self.n):
            #     if loop==0:
            #         optj=self.selectj(i,loop)
            #     elif self.alphas[i]<self.C and self.alphas[i]>0:
            #         optj=self.selectj(i,loop)
            #     else:
            #         continue
            #     if optj==None:
            #         continue
            #     alpha_new_unclip=self.alphas[optj]-(self.ei_alpha-self.ej_beta+2*self.tl)/self.ita(i,optj)
            #     alpha_new=self.clip_aa(i,optj,alpha_new_unclip)
            #     alphai_cache=self.alphas[i]
            #     alphaj_cache=self.alphas[optj]
            #     self.alphas[i]=-alpha_cache+alpha_new+self.alphas[optj]
            #     self.alphas[optj]=alpha_new
            #     self.updateb_aa(i,optj,alphai_cache,alphaj_cache)
            loop+=1
            self.w=(self.betas-self.alphas).T*self.cM
            print(self.w)
    def predict(self,x_test):
        y_predict=np.mat(x_test)*self.w.T+self.b
        return y_predict
        
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data=load_iris().data
target=load_iris().target
# data=np.zeros((100,4))
# target=np.zeros(100)
# for i in range(100):
#     data[i,:]=[i,10,i-10,i+20]
#     target[i]=i
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)
svr=SVR(1,0.05,200)
svr.fit(x_train,y_train)
y_p=svr.predict(x_train)
print(y_p)
print(y_train)
print(abs(y_p.T-y_train).mean())