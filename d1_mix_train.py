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
from sklearn.metrics.pairwise import euclidean_distances

RES_PATH = 'Results/'
DATA_PATH = 'Data/'

# argument parser
parser = argparse.ArgumentParser(description='D1 mix application')
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

Xtrain_scaled_path = os.path.join(RES_PATH, 'D1_mix_Xtrain_scaled.csv')
np.savetxt(Xtrain_scaled_path, xtrain_scaled, delimiter=",")


xtest_cent= X_test-X_train.mean(axis=0)
xtest_scaled=xtest_cent/xtrain_Centered.std(axis=0)

Xtest_scaled_path = os.path.join(RES_PATH, 'D1_mix_Xtest_scaled.csv')
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

def kernel_svm(C2,p1,gamma):
    def my_kernel(X,X1):
        p=np.shape(X)[1]
        L=np.array(diff_mat(n=64, m=64))
        D=L
        p=np.shape(X)[1]
        I_p=np.identity(p)
        a=I_p+C2*D.T@D
        A=np.linalg.inv(a)
        K=X@A@X1.T
        K=p1*K
        
        K1 = euclidean_distances(X, X1, squared=True)
        K1 *= -gamma
        K1=np.exp(K1, K1)
        K1=(1-p1)*K1
        
        # exponentiate K in-place
        
        return K+K1
    return my_kernel

# cross validation
print('-------- Cross Validation Start --------')
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import hinge_loss

# ======================
# Tune these grids
# ======================
C2_list    = [1e1, 1e-4]     # kernel parameter "C2"
gamma_list = [1e2, 1e3]      # kernel parameter "gamma"

# ======================
class CustomKernelSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, C2=1.0, gamma=0.2, probability=True, kernel_builder=None):
        self.C = C
        self.C2 = C2
        self.gamma = gamma
        self.probability = probability
        self.kernel_builder = kernel_builder
        self._svc_ = None
        self._p1_fixed = 0.6

    def fit(self, X, y):
        if self.kernel_builder is None:
            raise ValueError("kernel_builder must be provided")
        kern = self.kernel_builder(C2=self.C2, p1=self._p1_fixed, gamma=self.gamma)
        self._svc_ = SVC(C=self.C, kernel=kern, probability=self.probability)
        self._svc_.fit(X, y)
        return self

    def predict(self, X):
        return self._svc_.predict(X)

    def decision_function(self, X):
        return self._svc_.decision_function(X)

    def get_params(self, deep=True):
        return {
            "C": self.C,
            "C2": self.C2,
            "gamma": self.gamma,
            "probability": self.probability,
            "kernel_builder": self.kernel_builder,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

# ======================
# Label checks (Y must be in {-1, +1})
# ======================
classes, counts = np.unique(Y, return_counts=True)
if set(classes.tolist()) != {-1, 1}:
    raise ValueError(f"Y must be in {{-1,+1}}, got {classes}.")
min_count = counts.min()
if min_count < 2:
    raise ValueError(f"Minority class has only {min_count} sample(s); need ≥2 for stratified CV.")
k = min(5, int(min_count))
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# ======================
# Callable scorer: maximize negative hinge loss
# ======================
def neg_hinge_scorer(estimator, X_test, y_test):
    try:
        scores = estimator.decision_function(X_test)
    except Exception:
        scores = estimator.predict(X_test)
    return -hinge_loss(y_test, scores)

# ======================
# Pipeline & grid
# ======================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CustomKernelSVC(kernel_builder=kernel_svm, probability=True)),
])

param_grid = {
    "clf__C2": C2_list,
    "clf__gamma": gamma_list,
    # To also tune SVC's soft-margin C, add: "clf__C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=neg_hinge_scorer,
    cv=cv,
    n_jobs=-1,
    refit=True,
    error_score="raise",
)

# ======================
# Fit, extract best params, report
# ======================
gs.fit(X, Y)

# Save best params in variables named exactly as requested
gamma = gs.best_params_["clf__gamma"]
C2 = gs.best_params_["clf__C2"]

print("Overall best params:", gs.best_params_)
print(f"Best gamma: {gamma}")
print(f"Best C2: {C2}")
print("Min hinge loss (CV):", -gs.best_score_)

# --- Best C2 per gamma (summary) ---
res = gs.cv_results_
params_list = res["params"]
mean_scores = res["mean_test_score"]  # negative hinge loss

print("\nBest C2 per gamma (gamma, C2*, min_hinge_loss):")
for g in gamma_list:
    idxs = [i for i, p in enumerate(params_list) if p.get("clf__gamma") == g]
    if not idxs:
        continue
    best_i = max(idxs, key=lambda i: mean_scores[i])
    best_C2_for_g = params_list[best_i]["clf__C2"]
    best_loss_for_g = -mean_scores[best_i]
    print(f"{g:>10g}  {best_C2_for_g:>10g}  {best_loss_for_g:.6f}")

# Trained best model (Pipeline with scaler + CustomKernelSVC)
best_model = gs.best_estimator_

print('-------- Cross Validation End --------')


print(f'Optimal C2={C2}')
print(f'Optimal gamma={gamma}')

print('-------- Training --------')
clf = svm.SVC(kernel=kernel_svm(C2=C2, p1=0.6, gamma=gamma), probability=True)
clf=clf.fit(X,Y)
y_prob=clf.predict_proba(xtest_scaled)
y_pred1=clf.predict(xtest_scaled)


pred_path = os.path.join(RES_PATH, 'D1_mix_y_pred_aprl3.csv')
pred_proba_path = os.path.join(RES_PATH, 'D1_mix_y_pred_proba_aprl3.csv')
np.savetxt(pred_path, y_pred1, delimiter=",")
np.savetxt(pred_proba_path, y_prob, delimiter=",")


confusion_matrix = metrics.confusion_matrix(y, y_pred1)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix/len(Test1), display_labels = ['Neutral','Sweep'])
cm_display.plot()
cm_path = os.path.join(RES_PATH, 'D1_mix_cm_1000PC.pdf')
plt.savefig(cm_path)
plt.show()
print('accuracy score: %0.4f' % accuracy_score(y, y_pred1))

