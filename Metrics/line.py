# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
class line:
    
    def __init__(self,name):
        self.u = None
    
    def set_u(self, u):
        self.u = u
        
    def get_dist(q):
        return np.dot(u,q) 
    
