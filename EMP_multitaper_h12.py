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


RES_PATH = 'results_10kperclass/EMP/'
DATA_PATH = 'TFA/'


parser = argparse.ArgumentParser(description='application')
parser.add_argument('train_file', type=str, help='train file')
parser.add_argument('test_file', type=str, help='test file')

args = parser.parse_args()

# In[5]:


# filename = "/mnt/archive/home/aminm2021/Desktop/SISSSCO-main/MultiTaper_Sandy9/TFA//train_multitaper_9.csv"
filename = os.path.join(DATA_PATH, args.train_file)
df = pd.read_csv(filename) 


# In[6]:


df


# In[7]:


df.shape


# In[8]:


#df2=df.iloc[:, 8192:12288]


# In[9]:


df.shape


# In[10]:


# import seaborn as sns


# In[ ]:


# t2=np.mean(df2[10000:20000],axis=0)
# t2=t2.to_numpy()
# t2=t2.reshape(64, 64)
# plt.figure(figsize=(8,8))
# heat=sns.heatmap(t2)


# In[10]:


# import seaborn as sns
# df2=df[:9800]
# t2=np.mean(df2,axis=0)
# t2=t2.to_numpy()
# t2=t2.reshape(64, 64)


# In[11]:


# t2.min()


# In[12]:


# t2.max()


# In[13]:


# plt.figure(figsize=(8,8))
# heat=sns.heatmap(t2,vmax=1.1209364358037797,vmin=-0.7199181201590186)
# figure = heat.get_figure()    
#figure.savefig('/mnt/beegfs/home/aminm2021/Desktop/Trend-filtered SVM/heat_h12_sweep.pdf', dpi=400)


# In[14]:


#neut heat


# In[15]:


# import seaborn as sns
# df2=df[9800:19600]
# t2=np.mean(df2,axis=0)
# t2=t2.to_numpy()
# t2=t2.reshape(64, 64)


# In[16]:


# t2.min()


# In[17]:


# t2.max()


# In[18]:


# plt.figure(figsize=(8,8))
# heat=sns.heatmap(t2,vmax=1.1209364358037797,vmin=-0.5302000288486491)
# figure = heat.get_figure()    
#figure.savefig('/mnt/beegfs/home/aminm2021/Desktop/Trend-filtered SVM/heat_h12_neut.pdf', dpi=400)


# In[11]:


X_train=np.matrix(df)


# In[12]:


X_train


# In[13]:


X_train.shape


# In[ ]:


# #T1
# Test = np.empty((2000, 64, 64))

# for i in range(2000):
    
#     filename = "/mnt/beegfs/home/aminm2021/Desktop/Total_CEU_New/output" + str(i) + ".csv"
    
#     df = pd.read_csv(filename, index_col=0) 
#     df=np.matrix(df)
#     df=df.transpose()
#     df_Centered=df-df.mean(axis=0)
#     #df_scaled=df_Centered/df_Centered.std(axis=1)
#     df=df_Centered
#     df=(np.dot(df.transpose(),df))/64
#     df=pd.DataFrame(df)
#     Test[i] = df.to_numpy()       
#     print(i)


# In[15]:


# filename = "/mnt/archive/home/aminm2021/Desktop/SISSSCO-main/MultiTaper_Sandy9/TFA/test_multitaper_9.csv"
# df = pd.read_csv(filename)


# In[60]:


import numpy as np
import pandas as pd

# file_path = "/mnt/beegfs/home/aminm2021/Desktop/skinet/TFA/pirical_d_CEU22.csv_multitaper.npy"
file_path = os.path.join(DATA_PATH, args.test_file)

# load the numpy array
data = np.load(file_path)

# take the first 50 rows (works if data is 2D or more)
data_subset = data[:50]

# convert to DataFrame
df = pd.DataFrame(data_subset)

# save to CSV
# csv_path = "/mnt/beegfs/home/aminm2021/Desktop/skinet/TFA/pirical_d_CEU22_first50.csv"
csv_path = os.path.join(DATA_PATH, "pirical_d_CEU22.csv")
df.to_csv(csv_path, index=False)

print(f"Saved first 50 rows to: {csv_path}")


# In[129]:

filename = os.path.join(DATA_PATH, "pirical_d_CEU22.csv")
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


# In[94]:


#EMP
# filename = "/mnt/beegfs/home/aminm2021/Desktop/Trend-filtered SVM/Emp_Sandy/TFA/wavelet_chr_2.csv"
# df = pd.read_csv(filename)
# X_test=np.matrix(df)


# In[95]:


# X_test = Test.reshape(len(Test), -1)


# In[96]:


X_test.shape


# In[97]:


X_train


# In[98]:



neut=-np.ones(len(X_train)//2)
swp=np.ones(len(X_train)//2)
Y=np.concatenate((swp,neut))
# In[99]:


Y.shape


# In[100]:


# neut=-(np.ones(1000))
# swp=np.ones(1000)
# y=np.concatenate((swp,neut))


# In[ ]:





# In[101]:


# X_train = resize(
#     X_train, (X_train.shape[0] // 1, X_train.shape[1] // 9), anti_aliasing=True
# )


# In[102]:


xtrain_Centered=X_train-X_train.mean(axis=0)
xtrain_scaled=xtrain_Centered/xtrain_Centered.std(axis=0)
print(xtrain_Centered)
print(xtrain_scaled)

# In[103]:


xtrain_scaled.shape


# In[104]:


# tr_image_resized = resize(
#     xtrain_scaled, (xtrain_scaled.shape[0] // 1, xtrain_scaled.shape[1] // 9), anti_aliasing=True
# )


# In[105]:


# xtrain_scaled=tr_image_resized


# In[106]:


# X_train=xtrain_scaled


# In[107]:


# X_test.shape


# In[108]:

print(X_test.shape)
print(X_train.shape)
xtest_cent= X_test-X_train.mean(axis=0)
xtest_scaled=xtest_cent/xtrain_Centered.std(axis=0)


# In[41]:



# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean


# In[42]:


# tr_image_resized = resize(
#     xtrain_scaled, (xtrain_scaled.shape[0] // 1, xtrain_scaled.shape[1] // 9), anti_aliasing=True
# )


# In[43]:


# xtrain_scaled=tr_image_resized


# In[45]:


# test_image_resized = resize(
#     xtest_scaled, (xtest_scaled.shape[0] // 1, xtest_scaled.shape[1] // 9), anti_aliasing=True
# )


# In[46]:


# xtest_scaled=test_image_resized


# In[37]:


# import seaborn as sns


# In[ ]:


# tr_image_resized.


# In[41]:


# import seaborn as sns
# df2=tr_image_resized[:10000]
# t2=np.mean(df2,axis=0)
# #t2=t2.to_numpy()
# t2=t2.reshape(64, 64)


# In[42]:


# sns.heatmap(t2)


# In[43]:


# import seaborn as sns
# df2=tr_image_resized[10000:20000]
# t2=np.mean(df2,axis=0)
# #t2=t2.to_numpy()
# t2=t2.reshape(64, 64)
# sns.heatmap(t2)


# In[42]:


#D1


# In[109]:


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


# In[110]:


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


# In[111]:


# Fit custom kernel


# In[112]:


xtrain_scaled


# In[113]:


Y


# In[114]:
print(Y.shape)
print(xtrain_scaled.shape)

clf = svm.SVC(kernel=kernel_svm(C2=1e3),probability=True)
clf=clf.fit(xtrain_scaled,Y)


# In[131]:


y_pred=clf.predict_proba(xtest_scaled)


# In[ ]:


y_pred1=clf.predict(xtest_scaled)


# In[134]:

file_path = os.path.join(RES_PATH, 'D1_y_pred_proba_9_multi_12.csv')

# np.savetxt('/mnt/beegfs/home/aminm2021/Desktop/skinet/results_10kperclass/EMP/D1_y_pred_proba_9_multi_12.csv', y_pred, delimiter=",")
np.savetxt(file_path, y_pred, delimiter=",")
