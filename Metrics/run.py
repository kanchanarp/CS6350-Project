# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
import math
from Line import Line
from DistanceMetric import DistanceMetric
from Trajectory import Trajectory
    
def main():
    pts1 = [[1,2],[3,3],[2,1],[3,2],[5,5],[14,4]]
    pts2 = [[0,-2],[1,1],[4,1],[3,5],[7,5],[10,9]]
    lines1 = []
    for i in range(len(pts1)-1):
        l = Line(np.asarray(pts1[i]),np.asarray(pts1[i+1]))
        lines1.append(l)
    lines2 = []
    for i in range(len(pts2)-1):
        l = Line(np.asarray(pts2[i]),np.asarray(pts2[i+1]))
        lines2.append(l)
    t1 = Trajectory(lines1)
    t2 = Trajectory(lines2)
    Q = [np.array([0,0]),np.array([1,-2]),np.array([4,0]),np.array([12,-3])]
    d = DistanceMetric(Q)
    dist = d.calc_trajectorydst(Q,t1,t2)
    print(dist)
    #pass
        
if __name__=="__main__":main()