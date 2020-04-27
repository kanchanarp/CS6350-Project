#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import statistics as st
from io import open


# In[2]:


def convert_numeric(train, test, attr, convert_type):
    temp = [float(x[attr]) for x in train]

    if convert_type == 'median':
        a = st.median(temp)  
        
    elif convert_type == 'mean':
        a = st.mean(temp)
                                            
    for x in train:
        x[attr] = True if float(x[attr]) > a else False

    for x in test:
        x[attr] = True if float(x[attr]) > a else False


# In[3]:


def load_data(train_path, test_path, numeric_list = None, convert_type = 'median'):
    
    train = []
    train_labels = []
    with open(train_path, "r",encoding = 'utf-8-sig') as f:
        for line in f:
            item = line.strip().split(",")
            train.append(item[:-1])
            train_labels.append(item[-1])

    test = []
    test_labels = []
    with open(test_path, "r",encoding = 'utf-8-sig') as f:
        for line in f:
            item = line.strip().split(",")
            test.append(item[:-1])
            test_labels.append(item[-1])
    
    print(len(train))        
    if numeric_list == None:
        numeric_list = []
    
    else: 
        for attr in numeric_list:
            convert_numeric(train, test, attr, convert_type)

    return train, train_labels, test, test_labels


# In[4]:


import random
def my_sample(train, train_labels, n_samples = None):
    if n_samples == None:
        n_samples = len(train_labels)
    s_train = []
    s_lable = []
    for i in range(0, n_samples):
        x = random.randint(0, n_samples-1)
        s_train.append(train[x])
        s_lable.append(train_labels[x])

    return s_train, s_lable


# In[5]:


def Majority(labels, weights = None):
    
    
    if weights == None:
        L = len(labels)
        weights = [1]*L
    
    W = {}
    for x in range(len(labels)):
        
        if labels[x] not in W:
            W[labels[x]] = 0
            
        W[labels[x]] += weights[x]
    
    Max = -1
    majority = None    
    for y in W:
        if W[y] > Max:
            Max = W[y]
            majority = y
    return(majority, len(W))


# In[6]:


from math import log2
def entropy(labels, weights = None):
    
    n = len(labels)
    if weights == None:
        weights = [1]*n
            
    W = {}
    Sum = 0
    for i in range(n):
        if labels[i] not in W:
            W[labels[i]] = 0
        
        W[labels[i]] += weights[i]
            
        Sum += weights[i]
        
    S = 0
    for x in W:
        S += (W[x]/Sum) * log2(Sum / W[x])

    return S


# In[7]:


def Entropy_given_attribute(train, labels, attribute, weights = None):
    
    n = len(labels)
    if weights == None:
        weights = [1]*n
    
    
    split_l = {}
    split_w = {}
    sum_weights = sum(weights)
    
    for x in range(n):
        
        txa = train[x][attribute]
        if txa not in split_w:
            
            split_w[txa] =[]
            split_l[txa] = []
            
        split_w[txa].append(weights[x])
        split_l[txa].append(labels[x])  
        
    En = 0        
    for x in split_w:
        
        En += sum(split_w[x]) * entropy(split_l[x], split_w[x]) / sum_weights
        
    return(En, list(split_w.keys()))


# In[8]:


def old_best_att(train, labels, attributes, weights = None):
    
    
    lable_Ent = entropy(labels, weights)
    Max = -1
    Best = None
    Best_values = None
    
    for attribute in attributes: 
        temp, temp_values = Entropy_given_attribute(train, labels, attribute, weights) 
        if lable_Ent - temp >  Max:
            Max = lable_Ent - temp
            Best = attribute
            Best_values = temp_values
                    
    return(Best, Best_values)


# In[9]:


def split(train, label, attribute, weights = None):
    
    n = len(label)
    if weights == None:
        weights = [1]*n
    
    split_w = {}
    split_t = {}
    split_l = {}
    
    for x in range(len(label)):

        txa = train[x][attribute]
        if txa not in split_t:
            
            split_w[txa] = []
            split_t[txa] = []
            split_l[txa] = []
            
        split_w[txa].append(weights[x])
        split_t[txa].append(train[x])
        split_l[txa].append(label[x])
        
    return (split_t, split_l, split_w)


# In[10]:


def error(dt, x, y):
    count = 0
    for i in range(len(x)):
        xi = x[i]
        yi = dt.predict(xi)
        if yi != y[i]:
            count += 1

    return count / len(x)  


# In[11]:


class DecisionTree(object):
    def __init__(self, train, labels, attributes, depth = -1, weights = None):
        
        self.leaf = False 
        self.label, n_values = Majority(labels, weights) 
        
        if len(attributes) == 0 or n_values == 1 or depth == 0:
            
            self.leaf = True  
            return
        
        self.att_split, values = self.best_att(train, labels, attributes, weights)
        #print(self.att_split)
        
        train_s, lables_s, weight_s = split(train, labels, self.att_split, weights) #returns splited train, labels, weights as dicts
        
        self.Tree = {}
        
        attributes.remove(self.att_split)
        
        for v in train_s: # train_s is a dict whose keys are the values in column self.att_split
               
            self.Tree[v] = DecisionTree(train_s[v], lables_s[v], attributes, depth - 1, weight_s[v])

        attributes.append(self.att_split)
            
    
    def predict(self, instance):
        
        if self.leaf:
            return self.label
        
        if instance[self.att_split] in self.Tree:
            return self.Tree[instance[self.att_split]].predict(instance)   
        
        return self.label   
    
    
    def best_att(self, train, labels, attributes, weights):
        
        lable_Ent = entropy(labels, weights)
        Max = -1
        Best = None
        Best_values = None
    
        for attribute in attributes: 
            temp, temp_values = Entropy_given_attribute(train, labels, attribute, weights) 
            if lable_Ent - temp >  Max:
                Max = lable_Ent - temp
                Best = attribute
                Best_values = temp_values
                    
        return(Best, Best_values)


# In[30]:


#attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
# 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

attributes = [i for i in range(50)]
numeric_list = list(range(50))
    
train, labels, test, test_labels = load_data(
    "ID3_Train.csv", "ID3_Test.csv", numeric_list, convert_type = 'median')


# In[31]:


A = DecisionTree(train, labels, attributes, depth = -1, weights = None)


# In[32]:


P = []

for x in test:
    P.append(A.predict(x))


# In[33]:


print(P)
err = 0.0
for i in range(len(P)):
    err = err + (P[i]!=test_labels[i])
print(err/len(P))


# In[ ]:




