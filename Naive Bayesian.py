# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:02:02 2021

@author: PengTao
"""
import numpy as np
class BayesianRegression:
    def __init__(self,alpha=1000000):
        assert alpha>0,"alpha must not less than 0"
        self.alpha=alpha
        self.beta=None
        self.X_train=None
        self.Y_train=None
        self.mu=None
        self.sigma=None
    def fit(self,X_train,Y_train):
        self.X_train=X_train
        self.Y_train=Y_train
    def Gauss(self,X,Y):
        self.sigma=np.linalg.inv(np.dot(X.T,X)/self.beta+np.eye(X.shape[-1])/self.alpha)
        self.mu=np.dot(self.sigma,X.T).dot(Y)/self.beta
    def _predict(self,x_test):
        mean = np.dot(x_test,self.mu).sum()
        std = np.sqrt(np.dot(x_test,self.sigma).dot(x_test)).sum()
        return mean
    def predict(self,X_test):
        self.mle_variance()
        self.Gauss(self.X_train,self.Y_train)
        prediction=[self._predict(x) for x in X_test]
        return prediction
    def mle_variance(self):
        lr=LinearRegression(method="least_square")
        lr.fit(self.X_train,self.Y_train)
        self.beta=lr.evalue(self.X_train,self.Y_train)
    def evalue(self,X_test,Y_test):
        Y_p=self.predict(X_test)
        accuracy=np.square(Y_p-Y_test).mean()
        return accuracy
        
class LinearRegression:
    def __init__(self,method=None,lamda=None):
        if method==None:
            self.method="least_square"
        else:
            self.method=method
        self.method=method
        self.lamda=lamda
    def least_square(self,X,Y):
        """
        W=(X.T*X)^-1*X.T*Y
        """
        w=np.linalg.inv(np.dot(X.T,X)).dot(np.dot(X.T,Y))
        return w
    def ridge(self,X,Y,lamda):
        """
        W=(X.T*X+lamda*I)^-1*X.T*Y
        """
        assert lamda>0,"lamda must be not less than 0"
        w=np.linalg.inv(np.dot(X.T,X)+np.diag([lamda]*X.shape[-1])).dot(np.dot(X.T,Y))
        return w
    def fit(self,X,Y):
        self.X=X
        self.Y=Y
        return self
    def predict(self,X):
        if self.method=="least_square":
            w=self.least_square(self.X,self.Y)
            return np.dot(X,w)
        elif self.method=="ridge":
            if self.lamda==None:
                print("lamda not defined")
            else:
                w=self.ridge(self.X,self.Y,self.lamda)
            return np.dot(X,w)
        else:
            print("no such method called "+self.method)
    def evalue(self,X_test,Y_test):
        Y_p=self.predict(X_test)
        accuracy=np.square(Y_p-Y_test).mean()
        return accuracy
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data=load_iris().data
target=load_iris().target
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)

lr=LinearRegression(method="least_square")
lr.fit(x_train,y_train)
y_p=lr.predict(x_test)
acc=lr.evalue(x_test,y_test)
print(acc)
lrr=LinearRegression(method="ridge",lamda=0.5)
lrr.fit(x_train,y_train)
y_pr=lrr.predict(x_test)
accr=lrr.evalue(x_test,y_test)
print(accr)
br=BayesianRegression(alpha=0.001)
br.fit(x_train,y_train)
y_pb=br.predict(x_test)
accb=br.evalue(x_test,y_test)
print(accb)



