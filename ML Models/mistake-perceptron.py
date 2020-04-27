# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 05:27:57 2020

@author: Kanchana
"""
import math
import random
import numpy as np
from io import open

def read_file(filename,fnc):
    data = []
    count = 0
    with open(filename,'r',encoding = 'utf-8-sig') as f:
        for line in f:
            count = count + 1
            if(count > 0):
                terms = line.strip().split(',')
                data.append(np.asarray(list(map(fnc,terms))))
    return data

def sampleD(D,k):
    N = np.size(D,0)
    ids = random.sample(list(range(N)),k)
    ids.sort()
    D1 = np.array([np.array([0 for i in range(k)]) for j in range(k)])
    for i in range(k):
        for j in range(k):
            D1[i][j] = D[ids[i]][ids[j]]
    D2 = np.array([np.array([0 for i in range(N-k)]) for j in range(N-k)])
    ids_lft = []
    for i in range(N):
        if(not(i in ids)):
            ids_lft.append(i)
    for i in range(N-k):
        for j in range(N-k):
            D2[i][j] = D[ids_lft[i]][ids_lft[j]]
    ids_tst = random.sample(ids,int(k/2))+ids_lft
    ids_tst.sort()
    M = len(ids_tst)
    D3 = np.array([np.array([0 for i in range(k)]) for j in range(M)])
    for i in range(M):
        for j in range(k):
            D3[i][j] = D[ids_tst[i]][ids[j]]
    return D1,ids,D2,ids_lft,D3,ids_tst

def getY(usrs,inds):
    l_usr = [u[0] for u in usrs]
    usr_set = list(set(l_usr))
    usr_map={}
    for i in range(len(usr_set)):
        usr_map[usr_set[i]] = 2*i-1
    Y = []
    for i in range(len(l_usr)):
        if(i in inds):
            Y.append(usr_map[l_usr[i]])
    return Y

def toRBF(D,gamma=1):
    M = np.size(D,1)
    N = np.size(D,0)
    K = np.array([np.array([1 for j in range(M)]) for i in range(N)])
    for i in range(N):
        for j in range(M):
            K[i][j] = math.exp(-1*gamma*D[i][j]**2)
    return K

def mistake_perceptron(X,y,T):
    N = np.size(X,0)
    H = np.outer(y,y)*X
    c = np.array([0 for i in range(N)])
    idx = list(range(N))
    for i in range(T):
        random.shuffle(idx)
        for j in idx:
            if(np.dot(c,H[j])<=0):
                c[j] = c[j] + 1
    return c

def prediction_err(X,Y,Y_,c):
    N = np.size(X,0)
    pred = []
    cy = c*Y
    for i in range(N):
        pred.append(np.sign(np.dot(cy,X[i])))
    err = 0
    for j in range(N):
        err = err + (Y_[j]!=pred[j])
    err = 1.0*err / N
    return (pred,err)
    
def main():
    err_tr = []
    err_te = []
    T = 100
    usrs = read_file("Users_GeolifeXY.csv",str)
    D = read_file("Euclid_GeolifeXY.csv",float)
    D = toRBF(D,gamma = 1e-8)
    for j in range(T):
        #print('Test no: %d'%j)
        D1,inds1,D2,inds2,D3,indtst = sampleD(D,180)
        Y = getY(usrs,inds1)
        c = mistake_perceptron(D1,Y,100)
        Y_ = getY(usrs,indtst)
        pred,err = prediction_err(D1,Y,Y,c)
        err_tr.append(err)
        pred,err = prediction_err(D3,Y,Y_,c)
        err_te.append(err)
    print(sum(err_tr)/T)
    print(min(err_tr))
    print(max(err_tr))
    err_tr.sort()
    print((err_tr[T//2-1]+err_tr[T//2])/2)
    modes = []
    l = 0
    for e in err_tr:
        if(err_tr.count(e)>l):
            modes = [e]
            l = err_tr.count(e)
        elif(err_tr.count(e)==l):
            modes.append(e)
    print(set(modes))

    print(sum(err_te)/T)
    print(min(err_te))
    print(max(err_te))
    err_te.sort()
    print((err_te[T//2-1]+err_te[T//2])/2)
    modes = []
    l = 0
    for e in err_te:
        if(err_te.count(e)>l):
            modes = [e]
            l = err_te.count(e)
        elif(err_te.count(e)==l):
            modes.append(e)
    print(set(modes))
    
    
if __name__=="__main__":main()
            
    
