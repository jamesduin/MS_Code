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

fileName = open('results/rnd_results_fine','r')
fine_rnds = pickle.load(fileName)
fileName.close()

fileName = open('results/rnd_results_coarse','r')
coarse_rnds = pickle.load(fileName)
fileName.close()



f = open('test.txt', 'w')
results = []
f.write('coarse \n')
for i in range(1,int(coarse_rnds[-2][0])+1):
    roc_Sum = 0.0
    pr_Sum = 0.0
    acc_Sum = 0.0
    for row in coarse_rnds:
        if(row[0] == i):
            roc_Sum += row[2]
            pr_Sum += row[3]
            acc_Sum += row[4]
    f.write('{0},{1:.3f},{2:.3f},{3:.3f} \n'.format(i,(roc_Sum/10.0),(pr_Sum/10.0),(acc_Sum/10.0)))

f.write('fine \n')
for i in range(1,int(fine_rnds[-2][0])+1):
    roc_Sum = 0.0
    pr_Sum = 0.0
    acc_Sum = 0.0
    for row in fine_rnds:
        if(row[0] == i):
            roc_Sum += row[2]
            pr_Sum += row[3]
            acc_Sum += row[4]
    f.write('{0},{1:.3f},{2:.3f},{3:.3f} \n'.format(i,(roc_Sum/10.0),(pr_Sum/10.0),(acc_Sum/10.0)))

f.close()