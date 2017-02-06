import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
import time
import methods as m
import pickle

fileName1 = open('results/rnd_results_fine','rb')
fine_rnds = pickle.load(fileName1)
fileName1.close()

fileName2 = open('results/rnd_results_coarse','rb')
coarse_rnds = pickle.load(fileName2)
fileName2.close()



f = open('results/_rnds.txt', 'w')
results = []
f.write('{:^20}       {:^20}\n'.format('coarse','fine'))
for i in range(1,int(coarse_rnds[-3][0])+1):
    roc_Sum1 = 0.0
    pr_Sum1 = 0.0
    acc_Sum1 = 0.0
    for row1 in coarse_rnds:
        if(row1[0] == i):
            roc_Sum1 += row1[2]
            pr_Sum1 += row1[3]
            acc_Sum1 += row1[4]
    roc_Sum2 = 0.0
    pr_Sum2 = 0.0
    acc_Sum2 = 0.0
    for row2 in fine_rnds:
        if (row2[0] == i):
            roc_Sum2 += row2[2]
            pr_Sum2 += row2[3]
            acc_Sum2 += row2[4]
    f.write('{0:<4}{1:<5.3f},{2:<5.3f},{3:<5.3f}       '
            '{4:<4}{5:<5.3f},{6:<5.3f},{7:<5.3f}\n'.format(i,
            (roc_Sum1/10.0),(pr_Sum1/10.0),(acc_Sum1/10.0),
            i,(roc_Sum2 / 10.0), (pr_Sum2 / 10.0), (acc_Sum2 / 10.0)))
f.close()

###### Save coarse results to a file
f = open('results/_coarseResults.txt', 'w')
for result in coarse_rnds:
    if (type(result[2]) == str):
        f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
    elif (len(result) == 5):
        f.write('{0:<5}{1:<5}{2:<10.3f}{3:<10.3f}{4:<10.3f} \n'.format(*result))
    #else:
        #f.write(str(result).strip('[]'))
f.close()

###### Save results to a file
f = open('results/_fineResults.txt', 'w')
for result in fine_rnds:
    if (type(result[2]) == str):
        f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
    elif (len(result) == 5):
        f.write('{0:<5}{1:<5}{2:<10.3f}{3:<10.3f}{4:<10.3f} \n'.format(*result))
    #else:
        #f.write(str(result).strip('[]'))
f.close()