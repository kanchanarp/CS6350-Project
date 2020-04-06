# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:20:52 2020

@author: Kanchana
"""

import random
import numpy as np

def read_file(filename,biased = True):
    data = {'x':[],'y':[]}
    with open(filename,'r') as f:
        for line in f:
            terms = line.strip().split(',')
            flt_lst = list(map(float,terms[:-1]))
            if(biased):
                flt_lst.append(1)
            data['x'].append(np.asarray(flt_lst))
            data['y'].append(2*int(terms[-1])-1)
    data['x'] = np.asarray(data['x'])
    data['y'] = np.asarray(data['y'])
    return data

def gram_matrix(X,X_,k):
    sze1 = np.size(X,0)
    sze2 = np.size(X_,0)
    K = [[0 for i in range(sze1)] for j in range(sze2)]
    #print(np.size(K,1))
    for i in range(sze2):
        #print(i)
        for j in range(sze1):
            K[i][j] = k(X[j],X_[i])
    return K

def had_prodct(y,K):
    return np.outer(y,y)*K

gamma = 1
def kernel(x,y):
    global gamma
    return np.exp(-1*(np.linalg.norm(x-y)**2/gamma))
    
def kernel_perceptron_2(train,learn_rate,epochs,dim,k,biased = True):
    random.seed(1000)
    data = read_file(train,biased)
    #ss = []
    x = data['x']
    y = data['y']
    K = gram_matrix(x,x,k)
    H = np.outer(y,y)*K
    cnt = 0
    if(biased):
        dim = dim + 1
    w = np.asarray([0 for i in range(dim)])
    data_size = len(data['y'])
    c = np.array([0 for i in range(data_size)])
    for i in range(epochs):
        idx = list(range(data_size))
        random.shuffle(idx)
        for j in range(data_size):
            v = 0
            m =  idx[j]
            v = np.dot(H[m],c)
#            for r in range(data_size):
##                ll = k(x[r],x[m])
##                if(ll != K[r][m]):
##                    cnt = cnt + 1
#                v = v +learn_rate*c[r]*y[r]*y[m]*K[r][m]
            if(v <= 0):
                c[m] = c[m] + 1
    return c

def kernel_perceptron(X,Y,learn_rate,epochs,H):
    random.seed(1000)
    dim = np.size(X,1)
    data_size = np.size(Y)
    cnt = 0
    w = np.asarray([0 for i in range(dim)])
    c = np.array([0 for i in range(data_size)])
    for i in range(epochs):
        idx = list(range(data_size))
        random.shuffle(idx)
        for j in range(data_size):
            v = 0
            m =  idx[j]
            v = np.dot(H[m],c)
            if(v <= 0):
                c[m] = c[m] + 1
    return c

def give_predictions_err_2(test,c,X,Y,k,biased = True):
    data = read_file(test,biased)
    pred = []
    x = data['x']
    y = data['y']
    data_size = len(y)
    kern_size = np.size(c)
    K = gram_matrix(X,x,k)
    cy = c*Y
    for j in range(data_size):
        v = 0
        v = np.dot(cy,K[j])
#        for r in range(kern_size):
#            v = v + c[r]*Y[r]*k(X[r],x[j])
        pred.append(np.sign(v))  
    err = 0
    for j in range(data_size):
        err = err + (y[j]!=pred[j])
    err = 1.0*err / data_size
    return (pred,err)

def give_predictions_err(x,y,c,X,Y,K):
    pred = []
    data_size = len(y)
    kern_size = np.size(c)
    cy = c*Y
    for j in range(data_size):
        v = np.dot(cy,K[j])
        pred.append(np.sign(v))  
    err = 0
    for j in range(data_size):
        err = err + (y[j]!=pred[j])
    err = 1.0*err / data_size
    return (pred,err)

def main():
    g_lst = [0.1, 0.5, 1, 5, 100]
    global gamma
    data = read_file('bank-note/train.csv')
    X = data['x']
    Y = data['y']
    data = read_file('bank-note/test.csv')
    X_ = data['x']
    Y_ = data['y']
    r = 1
    T = 10
    dim = 4
    M = 1
    print("\\begin{tabular}{|c|c|c|} \\hline")
    print("$\gamma$ & Traning Error & Testing Error \\\\ \\hline \\hline")
    
    for g in g_lst:
        gamma = g
        err_tr = 0
        err_ts = 0
        M = 10
        k = kernel
        K = gram_matrix(X,X,k)
        H = had_prodct(Y,K)
        K_ = gram_matrix(X,X_,k)
        for i in range(M):
            #print(i)
            c = kernel_perceptron_2(X,Y,r,T,H)
            (_,err_t) = give_predictions_err_2(X,Y,c,X,Y,K)
            #print(err)
            (pred,err) = give_predictions_err_2(X_,Y_,c,X,Y,K_)
            err_tr = err_tr + err_t/M
            err_ts = err_ts + err/M
        print("%.1f & %.3f & %.3f\\\\ \\hline"%(g,err_tr,err_ts))
    print("\\end{tabular}")
if __name__=='__main__':main()