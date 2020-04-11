# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:04:18 2020

@author: Kanchana
"""
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def read_file(filename,biased = False):
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

def opt_func(a,K):
    return 0.5*np.dot(np.dot(K,a),a) - np.sum(a) 

def cons_func(a,y):
    return np.dot(a,y)

def cons_jac(a,y):
    return y

def opt_jac(a,K):
    return np.dot(a,K)-np.array([1 for i in a])

def find_b(a,K,y):
    b=0
    sze = np.size(y)
    for i in range(sze):
        b = b + y[i]-np.dot(a*y,K[i])
    return b/sze
    
def get_w(a,X,y):
    sze = np.size(y)
    dim = np.size(X,1)
    w = np.array([0 for i in range(dim)])
    for i in range(sze):
        w = w + a[i]*y[i]*X[i]
    return w

def get_pred_2(a,X,y,X_,k,b,y_):
    pred = []
    sze1 = np.size(X_,0)
    sze2 = np.size(y)
    for i in range(sze1):
        v = b
        for j in range(sze2):
            v = v + a[j]*y[j]*k(X_[i],X[j])
        pred.append(np.sign(v))
    err = 0
    for i in range(sze1):
        if(pred[i]!=y_[i]):
            err = err + 1
    return pred,1.0*err/sze1

def get_pred(a,y,K,b,y_):
    pred = []
    sze1 = np.size(y_)
    sze2 = np.size(y)
    for i in range(sze1):
        v = b + np.dot(a*y,K[i])
        pred.append(np.sign(v))
    err = 0
    for i in range(sze1):
        if(pred[i]!=y_[i]):
            err = err + 1
    return pred,1.0*err/sze1

gamma = 1
def kernel(x,y):
    global gamma
    #print(gamma)
    return np.exp(-1*(np.linalg.norm(x-y)**2/gamma))

def SVMdual(X,y,K,bnds,cons,x0):
    H = had_prodct(y,K)
    res = minimize(opt_func, x0, args = ([H]), method = 'SLSQP', bounds = bnds, constraints = cons, jac=opt_jac , options={'maxiter':10000})
    b=find_b(res.x,K,y)
    w=get_w(res.x,X,y)
    return res,w,b
    
def main():
    g_lst = [1]#[0.1, 0.5, 1, 5, 100]
    global gamma
    c_lst= [100,500,700]
    C= 1.0*100/873
    train = 'bank-note/train.csv'
    test = 'bank-note/test.csv'
    data = read_file(train)
    X = data['x']
    y = data['y']
    data = read_file(test)
    X_ = data['x']
    y_ = data['y']
    sze = np.size(y)
    #K = gram_matrix_2(X,X,k)
    #H = had_prodct(y,K)
    #K_ = gram_matrix_2(X,X_,k)
    cons = ({'type': 'eq', 'fun': cons_func,'jac': cons_jac, 'args': ([y])})
    #print("\\begin{tabular}{|c|c|c|c|} \\hline")
    #print("$C$ & Support Vectors &  Traning Error & Testing Error \\\\ \\hline \\hline")
    res_lst = []
    for c_ in c_lst:
        C = 1.0*c_/873
        bnds = [(0,C) for i in range(sze)]
        x0 = [C for i in range(sze)]     
        for g in g_lst:
            gamma = g
            k = np.dot
            K = gram_matrix(X,X,k)
            #H = had_prodct(y,K)
            K_ = gram_matrix(X,X_,k)
            #st = time.time()
            #res = minimize(opt_func_2, x0, args = (H), method = 'SLSQP', bounds = bnds, constraints = cons, jac=opt_jac , options={'maxiter':100000})
            #en = time.time()
            #print(res.x)
            #print("Time elapsed: %f"%(en-st))
            #b=find_b(res.x,K,y)
            #w=get_w(res.x,X,y)
            res,w,b = SVMdual(X,y,K,bnds,cons,x0)
            a = res.x
            n_spt = len(a[a!=0])
            res_lst.append(a)
            (_,err_tr) = get_pred(res.x,y,K,b,y)
            (_,err_ts) = get_pred(res.x,y,K_,b,y_)
            #print("$\\frac{%d}{%d}$ & %d & %f & %f\\\\ \\hline"%(c_,873,n_spt,err_tr,err_ts))
            print(w)
            print(b)
            print(err_tr)
            print(err_ts)
            
#    print("\\begin{tabular}{|c|c|c|} \\hline")
#    print("$\gamma_1$ & $\gamma_2$ & Number of Common Support Vectors \\\\")
#    for i in range(len(g_lst)):
#        for j in range(i+1,len(g_lst)):
#            l = res_lst[i]*res_lst[j]
#            n_spt = len(l[l>0])
#            print("%f & %f & %d \\\\ \\hline"%(g_lst[i],g_lst[j],n_spt))
#    print("\\hline \n \\end{tabular}")
if __name__=="__main__": main()

