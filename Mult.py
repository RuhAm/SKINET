import pandas as pd
import numpy as np
import os
from spectrum import *
import numpy as np
N=128    #### change this according to array size
NW=2.0
k=65
[tapers, eigen] = dpss(N, NW, k)
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser(description= 'Generate Time-frequency images for wavelet transform')
parser.add_argument('sweep_filename', type=str, help= 'Sweep summary statistic filename')
parser.add_argument('neutral_filename', type=str, help= 'Neutral summary statistic filename')
parser.add_argument('train', type=int, help= '# of Training samples')
parser.add_argument('test', type=int, help= '# of Test samples')
parser.add_argument('val', type=int, help= '# of validation samples')

args = parser.parse_args()

path1 = os.getcwd()

sw_file = args.sweep_filename
nt_file = args.neutral_filename

tr_n = args.train
ts_n = args.test
vl_n = args.val

sweep = pd.read_csv(path1 + '/Summary_statistics/' + sw_file ,header= None)
neutral = pd.read_csv(path1 + '/Summary_statistics/' + nt_file,header= None)

All_classes_train_xy = pd.concat([sweep.iloc[:tr_n, :], neutral.iloc[:tr_n, :]])
All_classes_val_xy = pd.concat([sweep.iloc[tr_n:tr_n +vl_n, :], neutral.iloc[tr_n:tr_n +vl_n, :]])
All_classes_test_xy= pd.concat([sweep.iloc[tr_n +vl_n :tr_n +vl_n +ts_n, :], neutral.iloc[tr_n+ vl_n:tr_n+vl_n+ts_n, :]])

label_col = ["Sweep"]*tr_n + ["Neutral"]*tr_n
All_classes_train_xy["Label"] = label_col
All_classes_train_xy= All_classes_train_xy.reset_index()
del All_classes_train_xy['index']
All_classes_train_xy = pd.get_dummies(All_classes_train_xy)
#All_classes_train_xy = All_classes_train_xy.sample(frac=1, random_state=0)


label_col = ["Sweep"]*vl_n + ["Neutral"]*vl_n
All_classes_val_xy["Label"] = label_col
All_classes_val_xy= All_classes_val_xy.reset_index()
del All_classes_val_xy['index']
All_classes_val_xy = pd.get_dummies(All_classes_val_xy)
#All_classes_val_xy = All_classes_val_xy.sample(frac=1, random_state=0)


label_col = ["Sweep"]*ts_n + ["Neutral"]*ts_n
All_classes_test_xy["Label"] = label_col
All_classes_test_xy= All_classes_test_xy.reset_index()
del All_classes_test_xy['index']
All_classes_test_xy = pd.get_dummies(All_classes_test_xy)
#All_classes_test_xy = All_classes_test_xy.sample(frac=1, random_state=0)



spec_tensor_train = np.empty((2*tr_n, 4096))
spec_tensor_val = np.empty((2*vl_n, 4096))
spec_tensor_test = np.empty((2*ts_n, 4096))


for i in range(2*tr_n):
 
 
  signal = list(np.array(All_classes_train_xy.iloc[i, :-2]))
  ###
  Sk_complex, weights, eigenvalues=pmtm(signal, e=eigen, v=tapers, NFFT=N, show=False)

  coefs = abs(Sk_complex)
  ###
  coefs_resized= np.array(resize(coefs, (64, 64)))
  coefs_flattened = coefs_resized.flatten()
  #print(coefs_flattened.shape)
  spec_tensor_train[i] = coefs_flattened
 
     

for i in range(2*ts_n):
 
 
  signal = list(np.array(All_classes_test_xy.iloc[i, :-2]))
  Sk_complex, weights, eigenvalues=pmtm(signal, e=eigen, v=tapers, NFFT=N, show=False)

  coefs = abs(Sk_complex)
  coefs_resized= np.array(resize(coefs, (64, 64)))
  coefs_flattened = coefs_resized.flatten()
  #print(coefs_flattened.shape)
  spec_tensor_test[i] = coefs_flattened
 
for i in range(2*vl_n):
 
 
  signal = list(np.array(All_classes_val_xy.iloc[i, :-2]))
  Sk_complex, weights, eigenvalues=pmtm(signal, e=eigen, v=tapers, NFFT=N, show=False)

  coefs = abs(Sk_complex)
  coefs_resized= np.array(resize(coefs, (64, 64)))
  coefs_flattened = coefs_resized.flatten()
  #print(coefs_flattened.shape)
  spec_tensor_val[i] = coefs_flattened
 
y_train = All_classes_train_xy.iloc[:, -2:]
y_test = All_classes_test_xy.iloc[:, -2:]
y_val = All_classes_val_xy.iloc[:, -2:]

y_train.to_csv('TFA/train_labels.csv', index= False)
y_test.to_csv('TFA/test_labels.csv', index= False)
y_val.to_csv('TFA/val_labels.csv', index= False)

spec_tensor_train = pd.DataFrame(spec_tensor_train)
spec_tensor_val = pd.DataFrame(spec_tensor_val)
spec_tensor_test = pd.DataFrame(spec_tensor_test)

spec_tensor_train.to_csv('TFA/train_sum_Mul_'+'.csv', index= False)
spec_tensor_test.to_csv('TFA/test_Mul_'+sw_file[:-4]+'.csv', index= False)
spec_tensor_val.to_csv('TFA/val_Mul_'+sw_file[:-4]+'.csv', index= False)
