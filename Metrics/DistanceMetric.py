# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
import math

class line:
    
    def __init__(self,u,st,en):
        self.u = u
        self.st = st
        self.en = en
    
    def set_u(self, u):
        self.u = u
        
    def set_st(self, st):
        self.st = st
        
    def set_u(self, en):
        self.en = en
        
    def get_dist(self,q):
        t = (1.0/np.linalg.norm(en-st)**2)*np.dot(q-st,en-st)
        t = min(max(t,0),1)
        pt = st+t*(en-st)
        dst = np.linalg.norm(q-pt)
        return (dst,pt) 
    
class Trjectory:
    
    def __init__(self,lines):
        self.lines = lines
        
    def set_lines(self,lines):
        self.lines = lines
        
    def get_lines(self):
        return self.lines
    
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
        D_a = calc_landmarkdst(Q,traj_a)
        D_b = calc_landmarkdst(Q,traj_b)
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

def main():
    pass
        
if __name__=="__main__":main()