# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:26:29 2020

@author: Kanchana
"""

import random
import numpy as np
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric


def calclateCost(Q,trajectories,centers):
    metric = DistanceMetric(Q)
    cluster_set = []
    cost = 0
    for t in trajectories:
        cnt  = None
        min_d = float('inf')
        for c in centers:
            d = metric.calc_trajectorydst(Q,t,c)[0]
            if(d>min_d):
                min_d = d
                cnt = c
        cluster_set.append(cnt)
        cost = cost + min_d
    return (cost,cluster_set)
            
def kMedoids(Q,trajectories,k,t_max):
    centers =  random.sample(trajectories, k)
    (cost,cluster_set) = calclateCost(Q,trajectories,centers)
    for i in range(t_max):
        tmp_cntrs = centers[:]
        cst_ = cost
        bst_ = tmp_cntrs
        for i in range(len(centers)):
            cst_swp=float('inf')
            best_swp = None
            for t in trajectories:
                if(not(t in centers)):
                    tmp_cntrs[i] = t
                    (cst,_) = calclateCost(Q,trajectories,centers)
                    if(cst<cst_swp):
                        cst_swp = cst
                        best_swp = tmp_centers[:]
            if(cst_swp < cst_):
                cst_ = cst_swp
                bst_ = bst_swp
        centers = bst_swp
    return centers
    
    