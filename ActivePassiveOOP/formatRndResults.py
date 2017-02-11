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
import methodsActPass as m
import pickle

fileName1 = open('results/active_fine','rb')
fine_rnds = pickle.load(fileName1)
fileName1.close()

fileName2 = open('results/active_coarse','rb')
coarse_rnds = pickle.load(fileName2)
fileName2.close()



f = open('results/_rnds.txt', 'w')
results = []
f.write('{:^20}       {:^20}\n'.format('coarse','fine'))
for i,result in enumerate(coarse_rnds):
    if(type(result[0]) != str and coarse_rnds[i][0] == fine_rnds[i][0]):
        f.write('{0:<4}{1:<5.3f},{2:<5.3f}        {3:<4}{4:<5.3f},{5:<5.3f}\n'.format(coarse_rnds[
                                                                                                              i][0],
                                                                                                          coarse_rnds[
                                                                                                              i][1],
                                                                                                          coarse_rnds[
                                                                                                              i][2],
                                                                                                          fine_rnds[i][
                                                                                                              0],
                                                                                                          fine_rnds[i][
                                                                                                              1],
                                                                                                          fine_rnds[i][
                                                                                                              2]))
f.close()
#
# ###### Save coarse results to a file
f = open('results/_coarseResults.txt', 'w')
for result in coarse_rnds:
    # if (type(result[2]) == str):
    #     f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
    # elif (len(result) == 5):
    #     f.write('{0:<5}{1:<10.3f}{2:<10.3f}{3:<10.3f} \n'.format(*result))
    #else:
    f.write(str(result) + '\n')
f.close()

###### Save results to a file
f = open('results/_fineResults.txt', 'w')
for result in fine_rnds:
    # if (type(result[2]) == str):
    #     f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
    # elif (len(result) == 5):
    #     f.write('{0:<5}{1:<10.3f}{2:<10.3f}{3:<10.3f} \n'.format(*result))
    #else:
    f.write(str(result)+'\n')
f.close()