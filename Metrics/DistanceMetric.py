# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
import math
    
class DistanceMetric:
    
    def __init__(self,Q):
        self.Q = Q
    
    def calc_landmarkdst(self,Q,trajectory):
        D = []
        for q in Q:
            min_dpt = None
            min_d = float('inf')
            for l in trajectory.get_lines():
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
            dist_Qpi = dist_Qpi + np.linalg.norm(D_a[i][1]-D_b[i][1])
        dist_Qpi = (1.0/n)*dist_Qpi
        return (dist_Q,dist_Qpi)