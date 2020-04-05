import numpy as np
import math
import time
import pandas as pd 
from scipy import linalg as LA
from scipy.optimize import minimize, Bounds



def to_float(data):
    for x in data:
        for i in range(len(data[0])):
            x[i] = float(x[i])
    return data



train1 = []

with open("bank-note/train.csv", "r") as f:
    for line in f:
        item = line.strip().split(",")
        train1.append(item) 
        
train2 = to_float(train1)        



test1 = []

with open("bank-note/test.csv", "r") as f:
    for line in f:
        item = line.strip().split(",")
        test1.append(item)
        
test2 = to_float(test1)

    

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



n = len(train)
m = len(test)
d = len(train[0]) - 1



train.shape, test.shape, d, n



X = train[:,:-1]
Y = train[:,-1]
y = train[:,-1:]




gamma_list = [0.1,0.5,1,5,100]
C_list = [100/873, 500/873, 700/873]



def K(x,z,gamma):
    return(math.exp((- LA.norm(x-z)**2)/gamma))



def Gram(A, B, gamma):  
    temp = np.sum(A**2,1).reshape(A.shape[0],1) + np.sum(B**2,1).reshape(1,B.shape[0])-2* A @ B.T
    return np.exp(-temp/gamma)



Ker = [np.zeros((n,n)) for i in range(len(gamma_list))]

for k in range(len(gamma_list)):
    Ker[k] = Gram(X,X,gamma_list[k])



SVM_kernel_dic = {}
a_dic = {}

start = time.time()
for j in range(len(gamma_list)):
    K1 = Gram(X, X, gamma_list[j])
    K2 = y*K1*y.T
    f = lambda x: 0.5 * x.T @ K2 @ x - np.sum(x) # x in place or a
    for i in range(len(C_list)):
        start = time.time()
        bounds = tuple([(0,C_list[i]) for k in range(n)])
        cons ={'type':'eq', 'fun': lambda x: x@Y}
        SVM_kernel_dic[(i,j)] = minimize(f, np.zeros(n), method='SLSQP', 
                                         bounds = bounds, constraints = cons) 
                                          # , options={'ftol': 1e-9, 'disp': True})
        a_dic[(i,j)] = SVM_kernel_dic[(i,j)].x
        print(time.time() - start)
print(time.time() - start)



a_list = np.zeros((len(C_list),len(gamma_list),n))

for i in range(len(C_list)):
    for j in range(len(gamma_list)):
        a_list[i][j] = a_dic[(i,j)]



supp_vec_alpha = np.zeros((len(C_list), len(gamma_list),n))

for k in range(len(gamma_list)):
    for i in range(len(C_list)):
        v = 0
        q = 0
        for j in range(n):
            if a_list[i][k][j] > 0: 
                u = j
                supp_vec_alpha[i][k][j] = a_list[i][k][u]



b_list = np.zeros((len(C_list), len(gamma_list)))

for k in range(len(gamma_list)):
    for i in range(len(C_list)):
        v = 0
        q = 0
        for j in range(n):
            if 1e-6 < a_list[i][k][j] < C_list[i] - 1e-8:
                v = v + 1
                q = q + Y[j]- sum(a_list[i][k][r]*Y[r]*Ker[k][r][j] for r in range(n))

        b_list[i][k] = q/v



def sgn(x):
    if x >=0:
        return 1
    else:
        return -1



def predict(x, gamma, C):
    
    for j in range(len(C_list)):
        if C == C_list[j]:
            l = j

    for k in range(len(gamma_list)):
        if gamma == gamma_list[k]:
            p = k 
     
    return sgn(sum(supp_vec_alpha[l][p][i]*Y[i]*K(X[i], x, gamma) for i in range(n))+ b_list[l][p])    



start = time.time()
label_train_predict = np.ones((len(C_list), len(gamma_list),n))

for j in range(len(C_list)):
    for k in range(len(gamma_list)):
        for i in range(n):
            label_train_predict[j][k][i] = predict(X[i],gamma_list[k], C_list[j])
            
print(time.time() - start)



start = time.time()
c = np.zeros((len(C_list),len(gamma_list)))

for j in range(len(C_list)):
    for k in range(len(gamma_list)):
        for i in range(n):
            if label_train_predict[j][k][i] != Y[i]:
                c[j][k] = c[j][k] + 1
print("Train error =", c/len(X))
print("Number of missclassified train examples:", c)
print(time.time() - start)



label_test_predict = np.ones((len(C_list),len(gamma_list),m))

for j in range(len(C_list)):
    for k in range(len(gamma_list)):
        for i in range(m):
            label_test_predict[j][k][i] = predict(test[i][:-1],gamma_list[k], C_list[j])
            


e = np.zeros((len(C_list),len(gamma_list)))

for j in range(len(C_list)):
    for k in range(len(gamma_list)):
        for i in range(m):
            if label_test_predict[j][k][i] != test[i][-1]:
                e[j][k] = e[j][k] + 1
print("Train error =", e/len(test))
print("Number of missclassified train examples:", e)



d1 = {}
d2 = {}
d3 = {}
Dic = [d1,d2,d3]

for i in range(len(C_list)):
    for k in range(len(gamma_list)):
        Dic[i][k+1] = [C_list[i]*(n+1), gamma_list[k], c[i][k]/n, e[i][k]/m, b_list[i][k]]



pd.DataFrame.from_dict(Dic[0], orient='index', 
                       columns=['873*C = 100','gamma','Train Error', 'Test Error', 'Biased'])



pd.DataFrame.from_dict(Dic[1], orient='index', 
                       columns=['873*C = 500','gamma','Train Error', 'Test Error', 'Biased'])



pd.DataFrame.from_dict(Dic[2], orient='index', 
                       columns=['873*C = 700','gamma','Train Error', 'Test Error', 'Biased'])


supp_vec = [[0] * len(gamma_list)] * len(C_list) 

I_list = []
count = np.zeros((len(C_list), len(gamma_list)))

for j in range(len(C_list)):
    for k in range(len(gamma_list)):
        I = [] 
        v = 0
        for i in range(n):
            if a_list[j][k][i] > 1e-6:
                I.append(i)
                v = v + 1
                
        I_list.append(I)
        count[j][k] = v
        supp_vec[j][k] = X[I, :]

count



I_list1 = [I_list[:5], I_list[5:10], I_list[10:]]



data = {'C = 100/873': count[0], 'C = 500/873': count[1], 'C = 700/873': count[2]}
pd.DataFrame.from_dict(data, orient='index', 
                       columns=['gamma = 0.1','gamma = 0.5','gamma = 1', 'gamma =5', 'gamma =100'])



Same = np.zeros(len(gamma_list)-1)

for k in range(len(gamma_list)-1):
    for i in range(len(I_list[5:10][k])):
        if I_list[:5][k][i] in I_list[5:10][k+1]:
            Same[k] += 1



S = list(Same)
S



dd = {'': ["Number of overlapped support vectors:"] + S}
pd.DataFrame.from_dict(dd, orient='index', 
                       columns=['(gamma_i, gamma_i+1):', '(0.1, 0.5)', 
                                '(0.5, 1)','(1, 5)', '(5, 10)'])

