# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:26:29 2020

@author: Kanchana
"""

import random
import numpy as np
import matplotlib.pyplot as plt
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
    
def calclateCost(Q,trajectories,centers):
    metric = DistanceMetric(Q)
    cluster_set = []
    cost = 0
    for t in trajectories:
        cnt  = None
        min_d = float('inf')
        for c in centers:
            d = metric.calc_euclideandst(t,c)
            if(d<min_d):
                min_d = d
                cnt = c
        cluster_set.append(cnt)
        cost = cost + min_d
    return (cost,cluster_set)
            
def kMedoids(Q,trajectories,k,t_max):
    centers =  random.sample(trajectories, k)
    print(centers)
    (cost,cluster_set) = calclateCost(Q,trajectories,centers)
    for i in range(t_max):
        print("Iteration: %d"%i)
        tmp_cntrs = centers[:]
        cst_ = cost
        bst_ = tmp_cntrs
        for i in range(len(centers)):
            cst_swp=float('inf')
            bst_swp = None
            for t in trajectories:
                if(not(t in tmp_cntrs)):
                    tmp_cntrs[i] = t
                    (cst,_) = calclateCost(Q,trajectories,tmp_cntrs)
                    if(cst<cst_swp):
                        cst_swp = cst
                        bst_swp = tmp_cntrs[:]
            if(cst_swp < cst_):
                cst_ = cst_swp
                bst_ = bst_swp
        centers = bst_
    return centers

def calcError(traj,traj_lst,lbls,cnt):
    clbls = {}
    lblvals = [-1,1]
    for i in range(len(cnt)):
        clbls[cnt[i]] = lblvals[i]
    error = 0
    for i in range(len(traj)):
        if(traj[i] in traj_lst['000']):
            obs = -1
        else:
            obs = 1
        c = lbls[i]
        pred = clbls[c]
        error = error + (pred!=obs)
    minerror = error
    clbls = {}
    lblvals = [1,-1]
    for i in range(len(cnt)):
        clbls[cnt[i]] = lblvals[i]
    error = 0
    for i in range(len(traj)):
        if(traj[i] in traj_lst['000']):
            obs = -1
        else:
            obs = 1
        c = lbls[i]
        pred = clbls[c]
        error = error + (pred!=obs)
    if(error<minerror):
        minerror = error
    return 1.0*minerror/len(traj)

def printDist(Q,trajectories):
    metric = DistanceMetric(Q)
    N = len(trajectories)
    for i in range(N):
        for j in range(N):
            if(i!=j):
                d = metric.calc_trajectorydst(Q,trajectories[i],trajectories[j])[1]
                if(d == float('inf')):
                    print("Traj: %d, %d"%(i,j))

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
    for j in range(T):
        print('Test no: %d'%j)
        idx = random.sample(list(range(sze)),10)  
        all_traj = []
        for i in idx:
            all_traj.append(traj_lst['000'][i])
        for i in idx:
            all_traj.append(traj_lst['001'][i])
        #all_traj = traj_lst['000'][idx] + traj_lst['001'][idx]
        #printDist(Q,all_traj)
        cnt = kMedoids(Q,all_traj,2,10)
        cst = calclateCost(Q,all_traj,cnt)
        err = calcError(all_traj,traj_lst,cst[1],cnt)
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
#    Q = [np.array([0,0]),np.array([1,-2]),np.array([4,0]),np.array([12,-3])]
#    d = DistanceMetric(Q)
#    dist = d.calc_trajectorydst(Q,t1,t2)
#    cnt = kMedoids(Q,[t1,t2,t3,t4],2,50)
#    print(cnt)
##    for t in cnt:
##        lines = t.get_lines()
##        print(t)
##        for l in lines:
##            print("Start:"+str(l.get_st())+" "+"End:"+str(l.get_en()))
#    cst = calclateCost(Q,[t1,t2,t3,t4],cnt)
#    colors = ['r','b']
##    trs = [t1,t2,t3,t4]
#    for i in range(len(all_traj)):
#        lines = all_traj[i].get_lines()
#        clr = colors[cnt.index(cst[1][i])]
#        for l in lines:
#            st = l.get_st()
#            en = l.get_en()
#            x = [st[0],en[0]]
#            y = [st[1],en[1]]
#            plt.plot(x,y,color=clr)
##    for p in Q:
#        plt.scatter(p[0],p[1],color='black')
#    #pass
        
if __name__=="__main__":main()