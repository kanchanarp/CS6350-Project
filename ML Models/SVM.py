# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:26:45 2020

@author: Kanchana
"""

import math
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from io import open
from sklearn import svm
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric

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

def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[i][0]==data[i+1][0] and data[i][1]==data[i+1][1])):
            l = Line(data[i],data[i+1])
            lines.append(l)
    return lines

def calcError(y,pred):
    error = 0
    N = len(y)
    for i in range(N):
        error = error + (y[i]!=pred[i])
    return 1.0*error/N

def getDist(Q,trajectories,method = 'paper1'):
    metric = DistanceMetric(Q)
    N = len(trajectories)
    D = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            if(i!=j):
                if(method=='paper1'):
                    d = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[1]
                if(method=='paper2'):
                    d = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[0]
                if(method=='euclid'):
                    d = metric.calc_euclideandst(trajectories[i],trajectories[j])
                if(d == float('inf')):
                    print("Traj: %d, %d"%(i,j))
                D[i][j] = d
            else:
                D[i][j] = 0
    return D

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
        usr_map[usr_set[i]] = i
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
            K[i][j] = math.exp(-1*gamma*D[i][j])
    return K
    
def main():
    random.seed(0)
#    files = ['000','001']
#    traj_lst = {'000':[] , '001':[]}
#    for name in files:
#        dirpath = "ExtractedData/"+name+"/CSV/*.csv"
#        dirlst = glob.glob(dirpath)
#        trajectories = []
#        lnths = []
#        for f in dirlst:
#            data = read_file(f)
#            lines = convertToLine(data)
#            if(len(lines)>100):
#                t = Trajectory(lines[:100])
#                trajectories.append(t)
#                lnths.append(len(lines))
#        lnths = np.asarray(lnths)
#        idx = lnths.argsort()[::-1]
#        traj = []
#        for i in idx:
#            traj.append(trajectories[i])
#        traj_lst[name] = traj
#    Q = []
#    N = 10
#    while(len(Q)<N):
#        x = random.randrange(437030,467040)
#        y = random.randrange(4416500,4436700)
#        p = [x,y]
#        if(not(p in Q)):
#            Q.append(p)
#    for i in range(len(Q)):
#        Q[i] = np.asarray(Q[i])
#    sze = min(len(traj_lst['000']),len(traj_lst['001']))
    err_ = []

    T = 10
    usrs = read_file("Users_GeolifeXY.csv",str)
    D = read_file("Euclid_GeolifeXY.csv",float)
    D = toRBF(D)
    for j in range(T):
        print('Test no: %d'%j)
        D1,inds1,D2,inds2,D3,indtst = sampleD(D,180)
        
        Y = getY(usrs,inds1)#[0 for i in range(M)]+[1 for i in range(M)]
        #np.array(getDist(Q,all_traj,method = 'paper2'))
        clf = svm.SVC(kernel = 'precomputed')
        clf.fit(D1,Y)
        #pred = clf.predict(D)
        #err = calcError(Y,pred)
#        idx = random.sample(list(range(sze)),M)  
#        all_traj = []
#        for i in idx:
#            all_traj.append(traj_lst['000'][i])
#        for i in idx:
#            all_traj.append(traj_lst['001'][i])
         #np.array(getDist(Q,all_traj,method = 'paper2'))
        
        pred = clf.predict(D1)
        err = calcError(Y,pred)
        Y = getY(usrs,indtst)
        err2 = clf.score(D3,Y)
        print(err)
        print(err2)
        err_.append(err)
    print(sum(err_)/T)
    print(min(err_))
    print(max(err_))
    err_.sort()
    print((err_[T//2-1]+err_[T//2])/2)
    modes = []
    l = 0
    for e in err_:
        if(err_.count(e)>l):
            modes = [e]
            l = err_.count(e)
        elif(err_.count(e)==l):
            modes.append(e)
    print(set(modes))
    
    
if __name__=="__main__":main()
