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



def createFolds(set, folds):
    for i in sorted(set):
        np.random.shuffle(set[i])
        partList = []
        for j in sorted(folds):
            partList.append((j, len(folds[j])))
        minIndex = partList[0][0]
        minVal = partList[0][1]
        for j in sorted(partList):
            if (minVal > j[1]):
                minVal = j[1]
                minIndex = j[0]
        partitionCounter = minIndex
        for instance in set[i]:
            folds[partitionCounter].append(instance)
            partitionCounter += 1
            if partitionCounter > 10:
                partitionCounter = 1