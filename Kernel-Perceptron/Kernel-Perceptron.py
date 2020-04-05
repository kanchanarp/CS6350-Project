#!/usr/bin/env python
# coding: utf-8

# # Processing Data

# In[17]:


import numpy as np
import pandas as pd 
import random 
import math
import time
from scipy import linalg as LA


# In[18]:


def to_float(data):
    for x in data:
        for i in range(len(data[0])):
            x[i] = float(x[i])
    return data


# In[19]:


train1 = []

with open("bank-note/train.csv", "r") as f:
    for line in f:
        item = line.strip().split(",")
        train1.append(item) #([1]+ item)
        
train2 = to_float(train1)        

for i in range(len(train1)):
    train1[i].insert(4,1)


test1 = []

with open("bank-note/test.csv", "r") as f:
    for line in f:
        item = line.strip().split(",")
        test1.append(item)
        
test2 = to_float(test1)

for i in range(len(test2)):
    test2[i].insert(4,1)
    

for i in range(len(train2)):
    train2[i] = np.array(train2[i])
    if train2[i][-1] < 0.5:
        train2[i][-1] = -1
    

train = np.array(train2)



for i in range(len(test2)):
    test2[i] = np.array(test2[i])
    if test2[i][-1] < 0.5:
        test2[i][-1] = -1


test = np.array(test2)


# In[20]:


n = len(train)
m = len(test)
d = len(train[0]) - 1


# In[21]:


train.shape, test.shape, d, n


# In[22]:


X = train[:,:-1]
Y = train[:,-1]
y = train[:,-1:]


# # Nonlinear Perceptron with Gaussian Kernel

# In[23]:


gamma_list = [0.1,0.5,1,5,100]


# In[24]:


def K(x,z,gamma):
    return(math.exp((- LA.norm(x-z)**2)/gamma))


# In[25]:


def Gram(A, B, gamma):  
    temp = np.sum(A**2,1).reshape(A.shape[0],1) + np.sum(B**2,1).reshape(1,B.shape[0])-2* A @ B.T
    return np.exp(-temp/gamma)


# In[26]:


start = time.time()
c = np.zeros((len(gamma_list),n))
T = 100

for k in range(len(gamma_list)):
    for t in range(T): 
        train_list = list(train)
        random.shuffle(train_list)
        train = np.array(train_list)
        G = Gram(train[:,:-1],train[:,:-1],gamma_list[k])
        for i in range(n):
            if train[:,-1][i] * ((c[k] * train[:,-1]) @ G[i]) <= 0:
                c[k][i] = c[k][i] + 1 

print(time.time() - start)
print(c)


# In[27]:


def sgn(x):
    if x >=0:
        return 1
    else:
        return -1


# In[28]:


def predict(Z,k):
    Q = sum(c[k].reshape(-1,1) * train[:,-1].reshape(-1,1) * Gram(train[:,:-1],Z,gamma_list[k]),0) # 0: add rows
    return np.where(Q >= 0, 1,-1)


# In[29]:


start = time.time()
a = np.zeros(len(gamma_list))

for k in range(len(gamma_list)):
    P = predict(X,k)
    for i in range(n):
        if P[i] * Y[i] < 0:
            a[k] = a[k] + 1
print("Train error =", a/n)
print("Number of missclassified train examples:", a)
print(time.time() - start)


# In[30]:


start = time.time()
b = np.zeros(len(gamma_list))

for k in range(len(gamma_list)):
    P = predict(test[:,:-1],k)
    for i in range(m):
        if P[i] * test[i][-1] < 0:
            b[k] = b[k] + 1
            
print("Test error =", b/m)
print("Number of missclassified test examples:", b)
print(time.time() - start)


# In[31]:


Dic = {}

for k in range(len(gamma_list)):
    Dic[k+1] = [gamma_list[k], a[k]/n, b[k]/m]


# In[32]:


pd.DataFrame.from_dict(Dic, orient='index', columns=['gamma','Train Error', 'Test Error'])


# # Without shuffeling with T=1

# In[487]:





# In[ ]:




