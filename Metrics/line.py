# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
class Line:
    
    def __init__(self,st,en):
        #self.u = u
        self.st = st
        self.en = en
    
    def set_u(self, u):
        self.u = u
        
    def set_st(self, st):
        self.st = st
        
    def set_u(self, en):
        self.en = en
        
    def get_dist(self,q):
        t = (1.0/np.linalg.norm(self.en-self.st)**2)*np.dot(q-self.st,self.en-self.st)
        t = min(max(t,0),1)
        pt = self.st+t*(self.en-self.st)
        dst = np.linalg.norm(q-pt)
        return (dst,pt)
    
