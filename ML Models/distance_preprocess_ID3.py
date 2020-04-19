# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:57:36 2020

@author: Kanchana
"""

import utm
import csv
import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import svm
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric

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

def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[i][0]==data[i+1][0] and data[i][1]==data[i+1][1])):
            l = Line(data[i],data[i+1])
            lines.append(l)
    return lines

Q = read_file("TDriveBest/best_points.csv")
print("File read!")
dirs = ['366','1131','2560','3015','3557','3579','6275','7146','8179','8717']
trajectories = []
for dr in dirs:
    dirpath = "TDriveBest/"+dr+"/*.csv"
    files = glob.glob(dirpath)
    print(dr)
    for f in files:
        data = read_file(f)
        lines = convertToLine(data)
        t = Trajectory(lines)
        t.set_user(int(dr))
        trajectories.append(t)
metric = DistanceMetric(Q)
data = []
print(len(trajectories))
for t in trajectories:
    print("%d %d"%(trajectories.index(t),len(t.get_lines())))
    D = metric.calc_landmarkdst(Q,t)
    D_ = [d[0] for d in D]
    D_.append(t.get_user())
    data.append(D_)
write_file("TDriveBest/ID3_Prep_2.csv",data)
#dirlst = glob.glob("*.txt")
#count = 0
#for file in dirlst:
#    data = read_file(file)
#    ids = set(data['date'])
#    dir_name = file.strip('.txt')
#    os.mkdir(dir_name)
#    for i in ids:
#        x = []
#        for j in range(len(data['date'])):
#            if(data['date'][j] == i):
#                x.append(data['x'][j])
#        filename = dir_name+"/"+str(i)+".csv"
#        write_file(filename,x)