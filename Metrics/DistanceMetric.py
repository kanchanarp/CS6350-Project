# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
import random
import math
    
class DistanceMetric:
    
    def __init__(self,Q):
        self.Q = Q
        self.D = None
    
    def calc_landmarkdst(self,Q,trajectory):
        D = []
        lines = trajectory.get_lines()
        for q in Q:
            min_dpt = None
            min_d = float('inf')
            for l in lines:
                (d,pt) = l.get_dist(q)  
                if(min_d>d):
                    min_d = d
                    min_dpt = pt
            D.append((min_d,min_dpt))
        return D
    
    def filt_lines(self,Q,trajectory):
        cl={}
        lines = trajectory.get_lines()
        for q in Q:
            lr = random.choice(lines)
            r = np.linalg.norm(q-lr.get_st())
            ql = []
            for l in lines:
                if(np.linalg.norm(q-l.get_st())<=r):
                    ql.append(l)
            cl[q] = ql
        return cl
        
    def calc_landmarkdst_opt(self,Q,trajectory):
        D = []
        cl = filt_lines(Q,trajectory)
        #lines = trajectory.get_lines()
        for q in Q:
            min_dpt = None
            min_d = float('inf')
            
            for l in cl[q]:
                (d,pt) = l.get_dist(q)  
                if(min_d>d):
                    min_d = d
                    min_dpt = pt
            D.append((min_d,min_dpt))
        return D
    
    def calc_trajectorydst(self,Q,traj_a,traj_b):
        D_a = self.calc_landmarkdst(Q,traj_a)
        D_b = self.calc_landmarkdst(Q,traj_b)
        n =  len(Q)
        dist_Q = 0
        for i in range(n):
            dist_Q = dist_Q + (D_a[i][0]-D_b[i][0])**2
        dist_Q = math.sqrt((1.0/n)*dist_Q)
        dist_Qpi = 0
        for i in range(n):
            #print(D_a[i][1])
            #print(D_b[i][1])
            dist_Qpi = dist_Qpi + np.linalg.norm(D_a[i][1]-D_b[i][1])
        dist_Qpi = (1.0/n)*dist_Qpi
        return (dist_Q,dist_Qpi)
    
    def calc_trajectorydst_opt(self,Q,traj_a,traj_b):
        D_a = self.calc_landmarkdst_opt(Q,traj_a)
        D_b = self.calc_landmarkdst_opt(Q,traj_b)
        n =  len(Q)
        dist_Q = 0
        for i in range(n):
            dist_Q = dist_Q + (D_a[i][0]-D_b[i][0])**2
        dist_Q = math.sqrt((1.0/n)*dist_Q)
        dist_Qpi = 0
        for i in range(n):
            #print(D_a[i][1])
            #print(D_b[i][1])
            dist_Qpi = dist_Qpi + np.linalg.norm(D_a[i][1]-D_b[i][1])
        dist_Qpi = (1.0/n)*dist_Qpi
        return (dist_Q,dist_Qpi)
    
    def calc_euclideandst(self,traj_a,traj_b):
        lines_a = traj_a.get_lines()
        lines_b = traj_b.get_lines()
        l = min(len(lines_a),len(lines_b))
        ed = 0
        for i in range(l):
            diff = lines_a[i].get_st()-lines_b[i].get_st()
            ed = ed + (1.0/l)*math.sqrt(np.dot(diff,np.transpose(diff)))
        return ed
    
    def calc_frechetdst(self,traj_a,traj_b):
        lines_a = traj_a.get_lines()
        lines_b = traj_b.get_lines()
        l = min(len(lines_a),len(lines_b))
        ed = 0
        for i in range(l):
            diff = lines_a[i].get_st()-lines_b[i].get_st()
            ed = ed + (1.0/l)*math.sqrt(np.dot(diff,np.transpose(diff)))
        return ed
    
    def DTW(i,j,c1,c2):
        if(i==0):
            v = np.linalg.norm(c1[0] - c2[j])
            self.D[j][i] = v
            return v
        if(j==0):
            v = np.linalg.norm(c1[i] - c2[0])
            self.D[j][i] = v
            return v
        if(self.D[j][i]!=-1):
            return self.D[j][i]
        v = np.linalg.norm(c1[i]-c2[j])+min(DTW(i-1,j,c1,c2),DTW(i-1,j-1,c1,c2),DTW(i,j-1,c1,c2))
        self.D[j][i] = v
        return v
        
    def calc_dtwdistance(self,traj_a,traj_b):
        lines_a = traj_a.get_lines()
        lines_b = traj_b.get_lines()
        k1 = len(lines_a)
        k2 = len(lines_b)
        c1 = []
        for l in lines_a:
            c1.append(l.get_st())
        c1.append(lines_a[-1].get_en())
        c2 = []
        for l in lines_b:
            c2.append(l.get_st())
        c2.append(lines_b[-1].get_en())
        self.D = [[-1 for i in range(k1+1)] for j in range(k2+1)]
        dst = DTW(k1,k2,c1,c2,self.D)
        return dst