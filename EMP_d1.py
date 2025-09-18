#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, datasets
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn import preprocessing


RES_PATH = 'Results/'
DATA_PATH = 'TFA/'


parser = argparse.ArgumentParser(description='application')
parser.add_argument('train_file', type=str, help='train file')
parser.add_argument('test_file', type=str, help='test file')

args = parser.parse_args()


filename = os.path.join(DATA_PATH, args.train_file)
df = pd.read_csv(filename) 


df.shape





X_train=np.matrix(df)


X_train





X_train.shape







import numpy as np
import pandas as pd


file_path = os.path.join(DATA_PATH, args.test_file)

# load the numpy array
data = np.load(file_path)

# take the first 50 rows (works if data is 2D or more)
data_subset = data[:50]

# convert to DataFrame
df = pd.DataFrame(data)

# save to CSV
# csv_path = "/mnt/beegfs/home/aminm2021/Desktop/skinet/TFA/pirical_d_CEU22_first50.csv"
csv_path = os.path.join(DATA_PATH, "empirical_CEU22.csv")
df.to_csv(csv_path, index=False)

print(f"Saved first 50 rows to: {csv_path}")


# In[129]:

filename = os.path.join(DATA_PATH, "empirical_CEU22.csv")
# filename = "/mnt/beegfs/home/aminm2021/Desktop/skinet/TFA/pirical_d_CEU22_first50.csv"
df = pd.read_csv(filename)


# In[130]:


df


# In[74]:


df2=df.iloc[:, 8192:12288]


# In[92]:


X_test=np.matrix(df2)


# In[93]:


X_test.shape
















X_test.shape





X_train






neut=-np.ones(len(X_train)//2)
swp=np.ones(len(X_train)//2)
Y=np.concatenate((swp,neut))



Y.shape







xtrain_Centered=X_train-X_train.mean(axis=0)
xtrain_scaled=xtrain_Centered/xtrain_Centered.std(axis=0)
print(xtrain_Centered)
print(xtrain_scaled)




xtrain_scaled.shape







print(X_test.shape)
print(X_train.shape)
xtest_cent= X_test-X_train.mean(axis=0)
xtest_scaled=xtest_cent/xtrain_Centered.std(axis=0)
















def diff_mat(n=64, m=64):
    p=n*m
    
    D_hor=np.zeros((n*(m-1),p))
    for j in range(1,n+1):
        for k in range(1,m):
            s=(j-1)*(m-1)+k
            D_hor[s-1,(j-1)*m+k-1]=1
            D_hor[s-1,(j-1)*m+k+1-1]=-1
    #l_hor=s

    D_vert=np.zeros((m*(n-1),p))
    for j in range(1,n):
        for k in range(1,m+1):
            s=(j-1)*m+k
            D_vert[s-1,(j-1)*m+k-1]=1
            D_vert[s-1,(j-1)*m+k+m-1]=-1
    #l_vert=s

    D_diag=np.zeros(((m-1)*(n-1),p))
    for j in range(1,n):
        for k in range(1,m):
            s=(j-1)*(m-1)+k
            D_diag[s-1,(j-1)*m+k-1]=1
            D_diag[s-1,(j-1)*m+k+m+1-1]=-1
    #l_diag=s
    
    D_Adiag=np.zeros(((m-1)*(n-1),p))
    for j in range(1,n):
        for k in range(1,m+1):
            s=(j-1)*(m-1)+k-1
            D_Adiag[s-1,(j-1)*m+k-1]=1
            D_Adiag[s-1,(j-1)*m+k+m-1-1]=-1
    #l_Adiag=s

    D=np.vstack((D_hor,D_vert,D_diag,D_Adiag))
    
    return D





def kernel_svm(C2):
    def my_kernel1(X,X1):
        p=np.shape(X)[1]
        L=np.array(diff_mat(n=64, m=64))
        D=L
        p=np.shape(X)[1]
        I_p=np.identity(p)
        a=I_p+C2*D.T@D
        A=np.linalg.inv(a)
        K=X@A@X1.T
        return K 
    return my_kernel1





# Fit custom kernel





xtrain_scaled





Y



print(Y.shape)
print(xtrain_scaled.shape)

clf = svm.SVC(kernel=kernel_svm(C2=1e3),probability=True)
clf=clf.fit(xtrain_scaled,Y)





y_pred=clf.predict_proba(xtest_scaled)





y_pred1=clf.predict(xtest_scaled)




file_path = os.path.join(RES_PATH, 'D1_y_pred_proba_9_multi_12.csv')

np.savetxt(file_path, y_pred, delimiter=",")
