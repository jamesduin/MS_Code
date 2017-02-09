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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
import time


def appendRndTimesFoldCnts(rndNum,results,sets):
    for lv in results:
        print('Round {0}: {1} seconds'.format(rndNum, round(time.perf_counter() - start_time, 2)))
        results[lv].append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
        instanceCount = 0
        fold_cnt = [lv+'_set']
        for i in sorted(sets[lv]):
            instanceCount += len(sets[lv][i])
            fold_cnt.append(['({0},{1})'.format(i, len(sets[lv][i]))])
        fold_cnt.append(['({0},{1})'.format('Total', instanceCount)])
        results[lv].append(fold_cnt)


def iterateFolds(rndNum, sets,test_part,results):
    for lvl in sets:
        results[lvl].append(['rnd', 'roc_auc', 'pr_auc','acc'])
        print("rnd" + str(rndNum) + "_" + lvl)

        ##### Create train set for coarse
        data = []
        for x in sorted(sets[lvl]):
            partition = np.asarray(sets[lvl][x])
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))

        y_train, X_train = data[:, 0], data[:, 1:data.shape[1]]
        y_trainCoarse = []
        for i in y_train:
            if i > 0:
                y_trainCoarse.append(1.)
            else:
                y_trainCoarse.append(i)
