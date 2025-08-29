"""
MD Ruhul Amin
aminm2021@fau.edu
Florida Atlantic University
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, datasets
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn import preprocessing
import argparse
from sklearn.model_selection import cross_val_score
from sklearn.metrics import hinge_loss
from sklearn import metrics

RES_PATH = 'Results/'
DATA_PATH = 'Data/'

# argument parser
parser = argparse.ArgumentParser(description='D1 application')
parser.add_argument('--train_obs', type=int, help='train sample number')
parser.add_argument('--test_obs', type=int, help='test sample number')
# parser.add_argument('--result_path', type=str, default=RES_PATH, help='result path')
# parser.add_argument('')

args = parser.parse_args()
print('argument parser: ', args)





max_range = len(os.listdir(os.path.join(DATA_PATH, 'Sweep')))
print(f'Total sweep files found: {max_range}')

# train test sampling
random.seed(123)
Train1 = random.sample(range(max_range), args.train_obs)
Train1.sort()
random.seed(1231)
Test1 = random.sample(range(max_range), args.test_obs)



# train
# Allocate arrays
Train_sweep = np.empty((len(Train1), 64, 64))
Train_neutral = np.empty((len(Train1), 64, 64))

# Read sweep files
for i in range(len(Train1)):
    filename = os.path.join(DATA_PATH, 'Sweep', "resized.2_" + str(Train1[i]) + ".csv")
    # filename = "/mnt/archive/home/aminm2021/Desktop/Trend-filtered SVM/CEU_2/CEU_2_May7/Cropped resized sweep/resized.2_" + str(Train1[i]) + ".csv"
    df = pd.read_csv(filename, index_col=0)
    Train_sweep[i] = df.to_numpy()
    if i % 10 == 9:
        print(f"Sweep {i + 1} / {len(Train1)}")

# Read neutral files
for i in range(len(Train1)):
    filename = os.path.join(DATA_PATH, 'Neutral', "resized.2_" + str(Train1[i]) + ".csv")
    # filename = "/mnt/archive/home/aminm2021/Desktop/Trend-filtered SVM/CEU_2/CEU_2_May7/Cropped resized neutral/resized.2_" + str(Train1[i]) + ".csv"
    df = pd.read_csv(filename, index_col=0)
    Train_neutral[i] = df.to_numpy()
    if i % 10 == 9:
        print(f"Neutral {i + 1} / {len(Train1)}")

# Merge sweep + neutral
Train = np.vstack((Train_sweep, Train_neutral))
print("Final Train shape:", Train.shape)


# Allocate arrays
Test_sweep = np.empty((len(Test1), 64, 64))
Test_neutral = np.empty((len(Test1), 64, 64))

# Read sweep files
for i in range(len(Test1)):
    filename = os.path.join(DATA_PATH, 'Sweep', "resized.2_" + str(Test1[i]) + ".csv")
    # filename = "/mnt/archive/home/aminm2021/Desktop/Trend-filtered SVM/CEU_2/CEU_2_May7/Cropped resized sweep/resized.2_" + str(Train1[i]) + ".csv"
    df = pd.read_csv(filename, index_col=0)
    Test_sweep[i] = df.to_numpy()
    if i % 10 == 9:
        print(f"Sweep {i + 1} / {len(Test1)}")

# Read neutral files
for i in range(len(Test1)):
    filename = os.path.join(DATA_PATH, 'Neutral', "resized.2_" + str(Test1[i]) + ".csv")
    # filename = "/mnt/archive/home/aminm2021/Desktop/Trend-filtered SVM/CEU_2/CEU_2_May7/Cropped resized neutral/resized.2_" + str(Train1[i]) + ".csv"
    df = pd.read_csv(filename, index_col=0)
    Test_neutral[i] = df.to_numpy()
    if i % 10 == 9:
        print(f"Neutral {i + 1} / {len(Test1)}")


# Merge sweep + neutral
Test = np.vstack((Test_sweep, Test_neutral))
print("Final Test shape:", Test.shape)


Train_updated = Train.reshape(len(Train), -1)
Test_updated = Test.reshape(len(Test), -1)

neut=-np.ones(len(Train1))
swp=np.ones(len(Train1))
Y=np.concatenate((swp,neut))

neut1=-np.ones(len(Test1))
swp1=np.ones(len(Test1))
y=np.concatenate((swp1,neut1))


X_train=Train_updated
X_test=Test_updated


xtrain_Centered=X_train-X_train.mean(axis=0)
xtrain_scaled=xtrain_Centered/xtrain_Centered.std(axis=0)

Xtrain_scaled_path = os.path.join(RES_PATH, 'D1Xtrain_scaled.csv')
np.savetxt(Xtrain_scaled_path, xtrain_scaled, delimiter=",")


xtest_cent= X_test-X_train.mean(axis=0)
xtest_scaled=xtest_cent/xtrain_Centered.std(axis=0)

Xtest_scaled_path = os.path.join(RES_PATH, 'D1Xtest_scaled.csv')
np.savetxt(Xtest_scaled_path, xtest_scaled, delimiter=",")

print('Standardization done')


X=xtrain_scaled
X_=xtest_scaled


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

# cross validation
print('-------- Cross Validation Start --------')
# C_list=[1e0,1e1,1e2,1e3]
C_list=[1e0,1e1]

val_loss=[]
for i in range(len(C_list)):
    clf = svm.SVC(kernel=kernel_svm(C2=C_list[i]))
    clf=clf.fit(X,Y)
    y_pred=clf.predict(X_)
    scores = cross_val_score(clf, X, Y, cv=5)
    loss=hinge_loss(y, y_pred)
    val_loss.append(loss)
    # print('accuracy score: %0.4f' % accuracy_score(y, y_pred))
    print(f'parameter: C2={i} | Validation loss = {loss}')

print('-------- Cross Validation End --------')
min_pos = np.argmin(val_loss)
C2 = C_list[min_pos] 

print(f'Optimal C2={C2}')

print('-------- Training --------')
clf=svm.SVC(kernel=kernel_svm(C2), probability=True)
clf=clf.fit(X,Y)
y_prob=clf.predict_proba(xtest_scaled)
y_pred1=clf.predict(xtest_scaled)


pred_path = os.path.join(RES_PATH, 'D1_y_pred_aprl3.csv')
pred_proba_path = os.path.join(RES_PATH, 'D1_y_pred_proba_aprl3.csv')
np.savetxt(pred_path, y_pred1, delimiter=",")
np.savetxt(pred_proba_path, y_prob, delimiter=",")


confusion_matrix = metrics.confusion_matrix(y, y_pred1)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix/len(Test1), display_labels = ['Neutral','Sweep'])
cm_display.plot()
cm_path = os.path.join(RES_PATH, 'D1_cm_1000PC.pdf')
plt.savefig(cm_path)
plt.show()
print('accuracy score: %0.4f' % accuracy_score(y, y_pred1))

