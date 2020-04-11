# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:57:36 2020

@author: Kanchana
"""

import utm
import csv
import glob
import os

def read_file(filename):
    data = []
    count = 0
    with open(filename,'r') as f:
        for line in f:
            count = count + 1
            if(count > 0):
                terms = line.strip().split(',')
                data.append(np.asarray(list(map(float,terms[:2]))))
    return data

def write_file(filename,data):
    f = open(filename,'w',encoding = 'utf-8-sig')
    lines = []
    for d in data:
        l = str(d[0])+","+str(d[1])+"\n"
        lines.append(l)
    f.writelines(lines)
    f.close()

def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[0][0]==data[1][0] and data[0][1]==data[1][1])):
            l = Line(data[0],data[1])
            lines.append(l)
    return lines

Q = read_file("BestQ.csv")
files = []
trajectories = []
for f in files:
    data = read_file(f)
    lines = convertToLine(data)
    t = Trajectory(lines)
    trajectories.append(t)
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