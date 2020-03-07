# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:26:45 2020

@author: Kanchana
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric

def read_file(filename):
    data = []
    count = 0
    with open(filename,'r') as f:
        for line in f:
            count = count + 1
            if(count > 6):
                terms = line.strip().split(',')
                data.append(np.asarray(list(map(float,terms[:2]))))
    return data

def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[0][0]==data[1][0] and data[0][1]==data[1][1])):
            l = Line(data[0],data[1])
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
                    d = metric.calc_trajectorydst(Q,trajectories[i],trajectories[j])[1]
                if(method=='paper2'):
                    d = metric.calc_trajectorydst(Q,trajectories[i],trajectories[j])[0]
                if(method=='euclid'):
                    d = metric.calc_euclideandst(trajectories[i],trajectories[j])
                if(d == float('inf')):
                    print("Traj: %d, %d"%(i,j))
                D[i][j] = d
            else:
                D[i][j] = 0
    return D

def main():
    random.seed(0)
    files = ['000','001']
    traj_lst = {'000':[] , '001':[]}
    for name in files:
        dirpath = "ExtractedData/"+name+"/CSV/*.csv"
        dirlst = glob.glob(dirpath)
        trajectories = []
        lnths = []
        for f in dirlst:
            data = read_file(f)
            lines = convertToLine(data)
            if(len(lines)>100):
                t = Trajectory(lines[:100])
                trajectories.append(t)
                lnths.append(len(lines))
        lnths = np.asarray(lnths)
        idx = lnths.argsort()[::-1]
        traj = []
        for i in idx:
            traj.append(trajectories[i])
        traj_lst[name] = traj
    Q = []
    N = 10
    while(len(Q)<N):
        x = random.randrange(437030,467040)
        y = random.randrange(4416500,4436700)
        p = [x,y]
        if(not(p in Q)):
            Q.append(p)
    for i in range(len(Q)):
        Q[i] = np.asarray(Q[i])
    sze = min(len(traj_lst['000']),len(traj_lst['001']))
    err_ = []
    T = 10
    M = min(20,sze)
    print(M)
    for j in range(T):
        print('Test no: %d'%j)
        idx = random.sample(list(range(sze)),M)  
        all_traj = []
        for i in idx:
            all_traj.append(traj_lst['000'][i])
        for i in idx:
            all_traj.append(traj_lst['001'][i])
        #all_traj = traj_lst['000'][idx] + traj_lst['001'][idx]
        #printDist(Q,all_traj)
        Y = [0 for i in range(M)]+[1 for i in range(M)]
        D = np.array(getDist(Q,all_traj,method = 'euclid'))
        clf = svm.SVC(kernel = 'precomputed')
        clf.fit(D,Y)
        #pred = clf.predict(D)
        #err = calcError(Y,pred)
        idx = random.sample(list(range(sze)),M)  
        all_traj = []
        for i in idx:
            all_traj.append(traj_lst['000'][i])
        for i in idx:
            all_traj.append(traj_lst['001'][i])
        D = np.array(getDist(Q,all_traj,method = 'euclid'))
        pred = clf.predict(D)
        err = calcError(Y,pred)
        print(err)
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