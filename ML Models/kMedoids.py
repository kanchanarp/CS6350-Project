# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:26:29 2020

@author: Kanchana
"""

import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from io import open
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric
##import similaritymeasures as sm
##from statistics import mode

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

def get_pts(t):
    lines = t.get_lines()
    pts = [lines[0].get_st()]
    for l in lines:
        pts.append(l.get_en())
    return pts

def get_mode(Y):
    S = list(set(Y))
    best = S[0]
    best_cont = 1
    for s in S:
        if(S.count(s)>best_cont):
            best = s
            best_cnt = S.count(s)
    return best
##def getDist(Q,trajectories,method = 'euclid'):
##    metric = DistanceMetric(Q)
##    N = len(trajectories)
##    D = [[0 for i in range(N)] for j in range(N)]
##    for i in range(N):
##        for j in range(N):
##            print("(%d , %d)"%(i,j))
##            if(i!=j):
##                if(method=='paper1'):
##                    d = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[1]
##                if(method=='paper2'):
##                    d = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[0]
##                if(method=='euclid'):
##                    d = metric.calc_euclideandst(trajectories[i],trajectories[j])
##                if(method == 'dtw'):
##                    pts1 = get_pts(trajectories[i])
##                    pts2 = get_pts(trajectories[j])
##                    d,_ = sm.dtw(pts1,pts2)
##                    #d = metric.calc_dtwdistance(trajectories[i],trajectories[j])
##                if(method=='frechet'):
##                    pts1 = get_pts(trajectories[i])
##                    pts2 = get_pts(trajectories[j])
##                    d = sm.frechet_dist(pts1,pts2)
##                    #d = metric.calc_fretchetdistance(trajectories[i],trajectories[j])
##                if(d == float('inf')):
##                    print("Traj: %d, %d"%(i,j))
##                D[i][j] = d
##            else:
##                D[i][j] = 0
##    return D

def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[i][0]==data[i+1][0] and data[i][1]==data[i+1][1])):
            l = Line(data[i],data[i+1])
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
            d = metric.calc_trajectorydst_opt(Q,t,c)[1]
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
        #print("Iteration: %d"%i)
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

def calclateCostOpt(trajectories,centers,D):
    #metric = DistanceMetric(Q)
    cluster_set = []
    cost = 0
    for i in range(len(trajectories)):
        #t = trajectories[i]
        cnt  = None
        min_d = float('inf')
        for j in range(len(centers)):
            d = D[i][centers[j]] #metric.calc_trajectorydst_opt(Q,t,c)[1]
            if(d<min_d):
                min_d = d
                cnt = j
        cluster_set.append(centers[cnt])
        cost = cost + min_d
    return (cost,cluster_set)

def kMedoidsOpt(trajectories,k,t_max,D):
    centers =  random.sample(range(len(trajectories)), k)
    #D = np.array(getDist(Q,trajectories,method = 'paper2'))
    (cost,cluster_set) = calclateCostOpt(trajectories,centers,D)
    for i in range(t_max):
        #print("Iteration: %d"%i)
        tmp_cntrs = centers[:]
        cst_ = cost
        bst_ = tmp_cntrs
        for i in range(len(centers)):
            cst_swp=float('inf')
            bst_swp = None
            for j in range(len(trajectories)):
                if(not(j in tmp_cntrs)):
                    tmp_cntrs[i] = j
                    (cst,_) = calclateCostOpt(trajectories,tmp_cntrs,D)
                    if(cst<cst_swp):
                        cst_swp = cst
                        bst_swp = tmp_cntrs[:]
            if(cst_swp < cst_):
                cst_ = cst_swp
                bst_ = bst_swp
        centers = bst_
    return centers

def calcError(traj,ids,lbls,cnt):
    lblvals = list(set(lbls))
    clstrs = {}
    slctd = []
    for l in lblvals:
        clstrs[l] = []
        for i in range(len(traj)):
            if(lbls[i]==l):
                clstrs[l].append(traj[i].get_user())
    err = 0.0
    for l in lblvals:
        cls = clstrs[l]
#        if(len(slctd)>0):
#            for s in slctd:
#                cls = list(filter((s).__ne__,cls))
        usr = get_mode(cls)
        err = err + 1.0*(len(clstrs[l])-cls.count(usr))
    return err/len(traj)
#    clbls = {}
#    lblvals = [-1,1]
#    for i in range(len(cnt)):
#        clbls[cnt[i]] = lblvals[i]
#    error = 0
#    for i in range(len(traj)):
#        if(traj[i] in traj_lst['000']):
#            obs = -1
#        else:
#            obs = 1
#        c = lbls[i]
#        pred = clbls[c]
#        error = error + (pred!=obs)
#    minerror = error
#    clbls = {}
#    lblvals = [1,-1]
#    for i in range(len(cnt)):
#        clbls[cnt[i]] = lblvals[i]
#    error = 0
#    for i in range(len(traj)):
#        if(traj[i] in traj_lst['000']):
#            obs = -1
#        else:
#            obs = 1
#        c = lbls[i]
#        pred = clbls[c]
#        error = error + (pred!=obs)
#    if(error<minerror):
#        minerror = error
#    return 1.0*minerror/len(traj)

def printDist(Q,trajectories):
    metric = DistanceMetric(Q)
    N = len(trajectories)
    for i in range(N):
        for j in range(N):
            if(i!=j):
                d = metric.calc_trajectorydst(Q,trajectories[i],trajectories[j])[1]
                if(d == float('inf')):
                    print("Traj: %d, %d"%(i,j))
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
    return D1,ids,D2,ids_lft

def variance(X):
    N = len(X)
    mu = 1.0*sum(X)/N
    var = 0.0
    for x in X:
        var = var + (1.0/N)*(x-mu)**2
    return var

def main():
    #random.seed(0)
    files = ['0','1','2','3','4','5','6','7','8','9']
    traj_lst = {'000':[] , '001':[]}
    all_traj = []
    for name in files:
        dirpath = "TDriveBest/XY/"+name+"/*.csv"
        dirlst = glob.glob(dirpath)
        trajectories = []
        lnths = []
        for f in dirlst:
            data = read_file(f,float)
            lines = convertToLine(data)
            if(len(lines)>100):
                t = Trajectory(lines)
                t.set_user(name)
                trajectories.append(t)
                lnths.append(len(lines))
        lnths = np.asarray(lnths)
        idx = lnths.argsort()[::-1]
        traj = []
        for i in idx:
            traj.append(trajectories[i])
        all_traj= all_traj+traj
    #all_traj = traj_lst['000'] + traj_lst['001']
#    Q = []
#    N = 10
#    while(len(Q)<N):
#        x = 39.0+2*random.random()
#        y = 115.0+2*random.random()
#        p = [x,y]
#        if(not(p in Q)):
#            Q.append(p)
#    for i in range(len(Q)):
#        Q[i] = np.asarray(Q[i])
#    sze = min(len(traj_lst['000']),len(traj_lst['001']))
    err_ = []
    print(len(all_traj))
    T = 10
    usrs = read_file("Users_TDrive.csv",str)
    D = read_file("Frechet_TDrive.csv",float)
    D1,inds1,D2,inds = sampleD(D,4)
    print(inds1)
    print(D1)
    for j in range(T):
        print('Test no: %d'%j)
        D1,inds1,D2,inds = sampleD(D,50)
        trajs = []
        usrs_ = []
        for i in inds1:
            #print(i)
            trajs.append(all_traj[i])
            usrs_.append(usrs[i])
#        idx = random.sample(list(range(sze)),10)  
#        #idx = list(range(sze))  
#        all_traj = []
#        for i in idx:
#            all_traj.append(traj_lst['000'][i])
#        for i in idx:
#            all_traj.append(traj_lst['001'][i])
#        #all_traj = traj_lst['000'][idx] + traj_lst['001'][idx]
        #printDist(Q,all_traj)
        #D = getDist(Q,all_traj,method = 'dtw')
#        L = []
#        for R in D[:4]:
#            L.append(R[:4])
#        print(L)
        cnt = kMedoidsOpt(trajs,2,100,D1)
        cst = calclateCostOpt(trajs,cnt,D1)
        #print(len(cnt))
        err = calcError(trajs,usrs_,cst[1],cnt)
        print(err)
        err_.append(err)
    print(sum(err_)/T)
    print(variance(err_))
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
