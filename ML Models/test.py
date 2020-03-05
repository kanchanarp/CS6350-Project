# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 05:07:51 2020

@author: Kanchana
"""
from Metrics.Line import Line
from Metrics.Trajectory import Trajectory
from Metrics.DistanceMetric import DistanceMetric
from sklearn import datasets

def main():
    iris_data = datasets.load_iris()
    print(iris_data.target)
    #pass
        
if __name__=="__main__":main()