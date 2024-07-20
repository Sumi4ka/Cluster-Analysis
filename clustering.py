import math
import numpy as np
from random import randint
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
K=3
#K=input() #число кластеров
iris = load_iris() #dataset IRIS
X = iris.data #numpy массив
Y = iris.target #numpy массив
N=len(Y)
def dist(a,b): #функция расстояния
    r=0
    for i in range(len(a)):
        r+=(a[i]-b[i])**2
    return math.sqrt(r)
def InitializationCentre(K):
    argCentre=np.empty(K)
    Centre=np.empty(K,dtype=np.ndarray)
    for i in range(K):
        argCentre[i]=randint(0,N-1)
        a=True
        while a:
            a=False
            for j in range(i):
                if argCentre[j]==argCentre[i]:
                    argCentre[i]=randint(0,N-1)
                    a=True
                    break
    for i in range(K):
        Centre[i]=X[int(argCentre[i])]
    return Centre
def Expectation(K,Centre):
    Y=np.empty(N)
    for i in range(N):
        mn=dist(Centre[0],X[i])
        argmn=0
        for j in range(K):
            if dist(Centre[j],X[i])<=mn:
                mn=dist(Centre[j],X[i])
                argmn=j
        Y[i]=argmn
    return Y
def Maximization(K,Y):
    Centre=np.empty(K,np.ndarray)
    for i in range(K):
        t=X.shape[1]
        A=np.empty(t+1,float)
        A.fill(0)
        for j in range(N):
            if Y[j]==i:
                for l in range(t):
                    A[l]+=X[j][l]
                A[t]+=1
        q=np.empty(t,float)
        if A[t]!=0:
            for l in range(t):
               q[l]=A[l]/A[t]
        else:
            q=A[0:t+1]
        Centre[i]=q
    return Centre
def Quality(K,Centre,Y):
    r=0
    for i in range(K):
        for j in range(N):
            if Y[j]==i:
                r+=dist(Centre[i],X[j])
    return r
def Plot(K,X,Y):
    Colour=np.array(['ro','ys','g^'])
    for k in range(K):
        for i in range(N):
            if Y[i]==k:
                plt.plot(X[i][0],X[i][2],Colour[k])
    
    plt.show()
M=15
r=1
R1=np.empty(M,np.ndarray)
R2=np.empty(M,float)
for m in range(M):
    Centre=InitializationCentre(K)
    NewCentre=Centre
    while r!=0:
        Centre=NewCentre
        Y=Expectation(K,Centre)
        NewCentre=Maximization(K,Y)
        a=np.empty(K)
        for i in range(K):
            a[i]=dist(NewCentre[i],Centre[i])
        r=max(a)
    R1[m]=NewCentre
    R2[m]=Quality(K,NewCentre,Expectation(K,NewCentre))
mn=R2[0]
argmn=0
for i in range(M):
    if R2[i]<mn:
        argmn=i
        mn=R2[i]
Plot(K,X,Expectation(K,R1[argmn]))