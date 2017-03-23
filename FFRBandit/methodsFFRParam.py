import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
import time
import pprint as pp
from decimal import *
import numpy as np
import random
getcontext().prec = 8

class LearnRound:
    def __init__(self,testFold,rndNum, lvl):
        self.lvl_rndTime = 0
        self.rndNum = rndNum
        self.testFold = testFold
        self.lvl = lvl


    def createTrainSet(self, set):
        self.lvl_rndTime = time.perf_counter()
        ##### Create train set for coarse
        data = []
        for x in sorted(set):
            partition = np.asarray(set[x])
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))

        y_train, X_train = data[:, 0], data[:, 1:]
        return y_train, X_train


    def printConfMatrix(self,y_testCoarse,y_predCoarse,results):
        ###### print conf matrix,accuracy and f1_score
        confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
        acc = accuracy_score(y_testCoarse, y_predCoarse)
        f1 = f1_score(y_testCoarse, y_predCoarse)
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['conf']+[confMatrix[0][0]] + [confMatrix[0][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['conf'] +[confMatrix[1][0]] + [confMatrix[1][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['acc']+[acc] +['f1']+[f1])


    def plotRocPrCurves(self,y_testCoarse,y_pred_score,y_sampleWeight,results):
        ###### Plot ROC and PR curves
        fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score, sample_weight=y_sampleWeight)
        roc_auc = auc(fpr, tpr, reorder=True)

        ##### Plog pr_curve
        precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_pred_score, sample_weight=y_sampleWeight)
        pr_auc = auc(recall, precision)
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['pr']+ [pr_auc]+ ['roc']+[roc_auc])
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['rndTime'] + [str(round(time.perf_counter() - self.lvl_rndTime, 2))])

class CoarseRound(LearnRound):
    def __init__(self,testFold,rndNum):
        LearnRound.__init__(self, testFold,rndNum, 'coarse')
        self.clf = []
        self.train_wt = 0.0

    def createTrainWtYtrain(self,y_train):
        ##### create train_wt and y_train for coarse
        y_trainCoarse = []
        for i in y_train:
            if i > 0:
                y_trainCoarse.append(1.0)
            else:
                y_trainCoarse.append(0.0)
        train_wt = fcnSclWeight(len(y_train) / np.sum(y_trainCoarse))
        self.train_wt = train_wt
        return y_trainCoarse

    def trainClassifier(self,X_train,y_trainCoarse):
        ##### Train classifier for coarse
        classifier = linear_model.LogisticRegression(penalty='l2',
                                                     C=0.1,
                                                     tol=0.00001,
                                                     solver='liblinear',
                                                     class_weight={1: self.train_wt},
                                                     n_jobs=-1)
        self.clf = classifier.fit(X_train, y_trainCoarse)


    def predictTestSet(self,X_test):
        ##### Predict test set for coarse
        y_predCoarse = self.clf.predict(X_test)
        y_pred_score = self.clf.decision_function(X_test)
        return y_predCoarse,y_pred_score

class FineRound(LearnRound):
    def __init__(self,testFold,rndNum):
        LearnRound.__init__(self, testFold,rndNum, 'fine')
        self.classifier = dict()
        self.Fine_wt = []

    def createTrainWtYtrain(self,y_train):
        ##### create train_wt (y_train unmodified) for fine
        y_trainBin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
        wt = len(y_train) / np.sum(y_trainBin)
        train_wt = fcnSclWeight(wt)
        self.Fine_wt = np.array(
            [3.0, 1.0, 1.0, 1.5,
             10.0, 2.0, 3.0, 1.0]) * train_wt
        return y_trainBin

    def trainClassifier(self,X_train,y_trainBin):
        #### train classifier for fine
        for cls in range(8):
            classif = linear_model.LogisticRegression(penalty='l2',
                                                         C=0.1,
                                                         tol=0.00001,
                                                         solver='liblinear',
                                                         class_weight={1: self.Fine_wt[cls]},
                                                         n_jobs=-1)
            clf = classif.fit(X_train, y_trainBin[:, cls])
            self.classifier[cls] = clf


    def predictTestSet(self,X_test):
        ##### predict test set for fine
        y_fine_score = []
        for cls in range(8):
            scores = self.classifier[cls].decision_function(X_test)
            scores = scores.reshape(scores.shape[0], 1)
            if y_fine_score == []:
                y_fine_score = scores
            else:
                y_fine_score = np.hstack((y_fine_score, scores))
        y_pred_score = np.amax(y_fine_score, axis=1)
        y_predCoarse = []
        for inst in y_pred_score:
            if (inst > 0.0):
                y_predCoarse.append(1.0)
            else:
                y_predCoarse.append(0.0)
        return np.array(y_predCoarse),y_pred_score




def fcnSclWeight(input):
    #return input
    #y = np.array([20.0, 6.5])
    y = np.array([23.0, 7.475])
    x = np.array([20.8870, 4.977])
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m * input + b






def createTestSet(test_part):
    ##### Create test set for coarse
    data_test = np.asarray(test_part)
    y_test, X_test = data_test[:, 0], data_test[:, 1:]
    y_testCoarse = []
    y_sampleWeight = []
    for inst in y_test:
        if inst > 0:
            y_testCoarse.append(1.0)
        else:
            y_testCoarse.append(0.0)
    test_wt = len(y_testCoarse) / np.sum(y_testCoarse)
    for inst in y_testCoarse:
        if inst > 0:
            y_sampleWeight.append(test_wt)
        else:
            y_sampleWeight.append(1.0)
    return y_testCoarse, y_sampleWeight, X_test




def predictCombined(results,y_pred_score,y_testCoarse,y_sampleWeight,rndNum,testFold):
    combPredScore = []
    combPredCoarse = []
    combPredLvl = []
    for i in range(len(y_testCoarse)):
        pred = max(y_pred_score['coarse'][i],y_pred_score['fine'][i])
        combPredScore.append(pred)
        if(pred == y_pred_score['coarse'][i]):
            combPredLvl.append('coarse')
        if(pred == y_pred_score['fine'][i]):
            combPredLvl.append('fine')
        if(pred > 0.0):
            combPredCoarse.append(1.0)
        else:
            combPredCoarse.append(0.0)

    ###### print conf matrix,accuracy and f1_score
    confMatrix = confusion_matrix(y_testCoarse, combPredCoarse)
    acc = accuracy_score(y_testCoarse, combPredCoarse)
    f1 = f1_score(y_testCoarse, combPredCoarse)
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['conf'] + [confMatrix[0][0]] + [confMatrix[0][1]])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['conf'] + [confMatrix[1][0]] + [confMatrix[1][1]])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['acc'] + [acc] + ['f1'] + [f1])
    combConfMat = dict()
    combConfMat['tn'] = ['tot',0,'coarse',0,'fine',0]
    combConfMat['fn'] = ['tot', 0, 'coarse', 0, 'fine', 0]
    combConfMat['fp'] = ['tot', 0, 'coarse', 0, 'fine', 0]
    combConfMat['tp'] = ['tot', 0, 'coarse', 0, 'fine', 0]
    for i in range(len(y_testCoarse)):
        if(y_testCoarse[i] == 0.0):
            if(y_testCoarse[i] == combPredCoarse[i]):
                combConfMat['tn'][1]+=1
                if(combPredLvl[i] == 'coarse'):
                    combConfMat['tn'][3] += 1
                if(combPredLvl[i] == 'fine'):
                    combConfMat['tn'][5] += 1
            else:
                combConfMat['fp'][1]+=1
                if(combPredLvl[i] == 'coarse'):
                    combConfMat['fp'][3] += 1
                if(combPredLvl[i] == 'fine'):
                    combConfMat['fp'][5] += 1
        if(y_testCoarse[i] == 1.0):
            if(y_testCoarse[i] == combPredCoarse[i]):
                combConfMat['tp'][1]+=1
                if(combPredLvl[i] == 'coarse'):
                    combConfMat['tp'][3] += 1
                if(combPredLvl[i] == 'fine'):
                    combConfMat['tp'][5] += 1
            else:
                combConfMat['fn'][1]+=1
                if(combPredLvl[i] == 'coarse'):
                    combConfMat['fn'][3] += 1
                if(combPredLvl[i] == 'fine'):
                    combConfMat['fn'][5] += 1
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['tn'] +combConfMat['tn'])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['fp'] +combConfMat['fp'])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['fn'] +combConfMat['fn'])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['tp'] +combConfMat['tp'])

    ###### Plot ROC and PR curves
    fpr, tpr, threshRoc = roc_curve(y_testCoarse, combPredScore, sample_weight=y_sampleWeight)
    roc_auc = auc(fpr, tpr, reorder=True)



    ##### Plot pr_curve
    precision, recall, threshPr = precision_recall_curve(y_testCoarse, combPredScore, sample_weight=y_sampleWeight)
    pr_auc = auc(recall, precision)
    addPrint(results,['fold']+[testFold]
                   +['comb_pr']+ [pr_auc]+ ['roc']+[roc_auc])


def retAddNum(add):
    remain = add % Decimal(1.0)
    num = int(add-remain)
    if(remain >0):
        x = Decimal(random.random())
        if(x < remain):
            num+= 1
    return num



def confEstAdd(results,classes_all,sets,rnds,add):
    decFcn = dict()
    for lvl in ['coarse', 'fine']:
        decFcn[lvl] = []

    for cls in sorted(classes_all):
        partition = np.asarray(classes_all[cls])
        data = partition
        if (len(data) > 0):
            y_train, X_train = data[:, 0], data[:, 1:]
            for lvl in ['coarse', 'fine']:
                y_predCoarse, scores = rnds[lvl].predictTestSet(X_train)
                s_tmp = np.abs(scores).reshape(scores.shape[0], 1)
                s_cls = np.ones(len(s_tmp)).reshape(len(s_tmp),1)*cls
                s_ind = np.array(range(len(s_tmp))).reshape(len(s_tmp),1)
                prefixCols = np.hstack((s_cls, s_ind))
                decFcnTmp = np.hstack((prefixCols,s_tmp))
                if decFcn[lvl] == []:
                    decFcn[lvl] = decFcnTmp
                else:
                    decFcn[lvl] = np.vstack((decFcnTmp,decFcn[lvl]))
    for lvl in ['coarse', 'fine']:
        d_ind = np.array(range(len(decFcn[lvl]))).reshape(len(decFcn[lvl]), 1)
        decFcn[lvl] = np.hstack((d_ind,decFcn[lvl]))
        decFcn[lvl] = decFcn[lvl][decFcn[lvl][:,3].argsort()].tolist()

    addFine = retAddNum(add['fine'])
    addCoarse = retAddNum(add['coarse'])
    removeInd = []
    for i in range(addFine):
        most_cls = int(decFcn['fine'][i][1])
        most_ind = int(decFcn['fine'][i][2])
        removeInd.append([most_cls,most_ind])
        sets['coarse'][most_cls].append(classes_all[most_cls][most_ind])
        sets['fine'][most_cls].append(classes_all[most_cls][most_ind])
        # addPrint(results,['fine']+['cls']+[most_cls]+['ind']+
        #      [most_ind]+['mostUncert']+[most_uncert]+['coarseUncert']+[coarseUncert])

    for i in range(addCoarse):
        most_cls = int(decFcn['coarse'][i][1])
        most_ind = int(decFcn['coarse'][i][2])
        while([most_cls,most_ind] in removeInd):
            del decFcn['coarse'][i]
            try:
                most_cls = int(decFcn['coarse'][i][1])
                most_ind = int(decFcn['coarse'][i][2])
            except IndexError:
                print("ran out of coarse indexes")
        removeInd.append([most_cls,most_ind])
        sets['coarse'][most_cls].append(classes_all[most_cls][most_ind])
        # addPrint(results,['coarse']+['cls']+[most_cls]+['ind']+
        #          [most_ind]+['mostUncertCoarse']+[most_uncert]+['fineUncert']+[finUncert])
    addPrint(results,['addFine']+[addFine]+['addCoarse']+[addCoarse])

    removeInd = np.array(removeInd)
    removeInd = removeInd[removeInd[:,1].argsort()[::-1]]
    if(len(removeInd)!=addCoarse+addFine):
        addPrint(results,"Didn't add expected amount to coarse and fine sets.")
        raise SystemExit
    for i,inst in enumerate(removeInd):
        most_cls = int(removeInd[i][0])
        most_ind = int(removeInd[i][1])
        del classes_all[most_cls][most_ind]



def appendSetTotal(rndNum, results, sets,name):
    instanceCount = 0
    fold_cnt = ['rnd'] + [rndNum] + [name]
    for i in sorted(sets):
        instanceCount += len(sets[i])
        fold_cnt.append((i, len(sets[i])))
    fold_cnt.append(('tot', instanceCount))
    addPrint(results, fold_cnt)


def appendRndTimesFoldCnts(testFold, rndNum, results, sets, start_time):
    instanceCount = dict()
    for lvl in ['coarse', 'fine']:
        instanceCount[lvl] = 0
        fold_cnt = ['rnd']+[rndNum] +['fold']+[testFold]+['lvl']+[lvl]
        for i in sorted(sets[lvl]):
            instanceCount[lvl] += len(sets[lvl][i])
            fold_cnt.append((i, len(sets[lvl][i])))
        fold_cnt.append(('tot', instanceCount[lvl]))
        addPrint(results,fold_cnt)
    addPrint(results,['rnd']+[rndNum] +['fold']+[testFold]
                   +['rndTimeTot'] + [str(round(time.perf_counter() - start_time[-1], 2))])
    return max([instanceCount['fine'],instanceCount['coarse']])







def switchClass5instance(test_part,train_part):
    for i, inst in enumerate(test_part):
        if inst[0] == 5:
            for part in train_part:
                train_part[part].append(test_part.pop(i))
                return




def printClsVsFolds(results,folds, title):
    addPrint(results,'{:<14}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(title, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    instanceCount = 0
    classCountTot = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in sorted(folds):
        classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        instanceCount += len(folds[i])
        for inst in folds[i]:
            classCountTot[int(inst[0])] += 1
            classCount[int(inst[0])] += 1
        classCount = [i] + [len(folds[i])] + classCount
        addPrint(results,'{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))
    addPrint(results,'{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total', instanceCount, *classCountTot))
    return classCount


def printClassTotals(results,classes):
    addPrint(results,'{0:<10}{1:<10}'.format('Classes', ''))
    instanceCount = 0
    for i in sorted(classes):
        instanceCount += len(classes[i])
        addPrint(results,'{0:<10}{1:<10}'.format(i, len(classes[i])))
    addPrint(results,'{0:<10}{1:<10}\n'.format('Total', instanceCount))
    addPrint(results,'Shape: {0:<10}\n'.format(len(classes[0][0])))


def addPrint(results,x):
    results.append(x)
    print(x)



def loadScaledPartData(dataDir):
    all_part = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for i in sorted(all_part):
        with open(dataDir + 'data/partitionMinMaxScaled/partitionMinMaxScaled_' + str(i)) as f:
        #with open(dataDir + 'data/partition_scaled/partition_scaled' + str(i)) as f:
            for line in f:
                nums = line.split()
                nums = list(map(float, nums))
                all_part[i].append(nums)
        np.random.shuffle(all_part[i])
    return all_part




