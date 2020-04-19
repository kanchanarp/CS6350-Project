# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:01:46 2020

@author: Kanchana
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:26:29 2020

@author: Kanchana
"""

import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric
import similaritymeasures as sm

def read_file(filename):
    data = []
    count = 0
    with open(filename,'r',encoding = 'utf-8-sig') as f:
        for line in f:
            count = count + 1
            if(count > 0):
                terms = line.strip().split(',')
                data.append(np.asarray(list(map(float,terms[:2]))))
    return data

def write_file(filename,data):
    f = open(filename,'w',encoding = 'utf-8-sig')
    lines = []
    #sze = np.size(data,1)
    for d in data:
        st = list(map(str,d))
        l = ","
        l = l.join(st)
        l = l + "\n"
        lines.append(l)
    f.writelines(lines)
    f.close()
    
def get_pts(t,bnd):
    lines = t.get_lines()
    n = int(len(lines)/bnd)
    pts = [lines[0].get_st()]
    for i in range(0,len(lines),n):
        pts.append(lines[i].get_en())
    return pts
    
def getDist(Q,trajectories,method = 'euclid'):
    metric = DistanceMetric(Q)
    N = len(trajectories)
    D = [[0 for i in range(N)] for j in range(N)]
    D1 = [[0 for i in range(N)] for j in range(N)]
    D2 = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            print("%s \t - \t (%d , %d)"%(method,i,j))
            if(i!=j):
                if(method=='paper1'):
                    d1 = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[1]
                if(method=='paper2'):
                    d2 = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[0]
                if(method=='euclid'):
                    d = metric.calc_euclideandst(trajectories[i],trajectories[j])
                if(method == 'dtw'):
                    ti = trajectories[i]
                    tj = trajectories[j]
                    bnd = min(len(ti.get_lines()),len(tj.get_lines()),200)
                    pts1 = get_pts(ti,bnd)
                    pts2 = get_pts(tj,bnd)
                    d,_ = sm.dtw(pts1,pts2)
                    #d = metric.calc_dtwdistance(trajectories[i],trajectories[j])
                if(method=='frechet'):
                    ti = trajectories[i]
                    
                    tj = trajectories[j]
                    bnd = min(len(ti.get_lines()),len(tj.get_lines()),100)
                    pts1 = get_pts(ti,bnd)
                    pts2 = get_pts(tj,bnd)
                    d = sm.frechet_dist(pts1,pts2)
                    #d = metric.calc_fretchetdistance(trajectories[i],trajectories[j])
                if(d == float('inf')):
                    print("Traj: %d, %d"%(i,j))
                D[i][j] = d
                
            else:
                D[i][j] = 0
    return D

def getDistPapr(Q,trajectories):
    metric = DistanceMetric(Q)
    N = len(trajectories)
    D1 = [[0 for i in range(N)] for j in range(N)]
    D2 = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(i,N):
            print("Paper \t - \t (%d , %d)"%(i,j))
            if(i!=j):
                r = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])
                d1 = r[1]
                d2 = r[0]
                if(d1 == float('inf') or d2 == float('inf')):
                    print("Traj: %d, %d"%(i,j))
                D1[i][j] = d1
                D1[j][i] = d1
                D2[i][j] = d2
                D2[j][i] = d2
            else:
                D1[i][j] = 0
                D2[i][j] = 0
    return D1,D2
def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[i][0]==data[i+1][0] and data[i][1]==data[i+1][1])):
            l = Line(data[i],data[i+1])
            lines.append(l)
    return lines    

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
                t = Trajectory(lines)
                t.set_user(name)
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
        x = 39.0+2*random.random()
        y = 115.0+2*random.random()
        u1,u2,_,_ = utm.from_latlon(x,y)
        p = [u1,u2]
        if(not(p in Q)):
            Q.append(p)
    for i in range(len(Q)):
        Q[i] = np.asarray(Q[i])
    sze = min(len(traj_lst['000']),len(traj_lst['001']))
    err_ = []
    T = 1
    print("Writing Q")
    write_file("Q_GeolifeXY.csv",Q)
    
    print("Writing users")
    usrs = ['000']*len(traj_lst['000'])+['001']*len(traj_lst['001'])
    write_file("Users_GeolifeXY.csv",usrs)
    
    print("DTW method")
    all_traj = traj_lst['000'] + traj_lst['001']
    D = getDist(Q,all_traj,method = 'dtw')
    write_file("DTW_GeolifeXY.csv",D)
    
    print("Frechet method")
    all_traj = traj_lst['000'] + traj_lst['001']
    D = getDist(Q,all_traj,method = 'frechet')
    write_file("Frechet_GeolifeXY.csv",D)
    
    print("Paper method")
    all_traj = traj_lst['000'] + traj_lst['001']
    D1,D2 = getDistPapr(Q,all_traj)
    write_file("Paper1_GeolifeXY.csv",D1)
    write_file("Paper2_GeolifeXY.csv",D2)
    
    
        
if __name__=="__main__":main()