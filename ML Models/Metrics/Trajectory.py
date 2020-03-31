# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:17:42 2020

@author: Kanchana
"""
import numpy as np
import math
    
class Trajectory:
    
    def __init__(self,lines):
        self.lines = lines
        
    def set_lines(self,lines):
        self.lines = lines
        
    def get_lines(self):
        return self.lines
    
