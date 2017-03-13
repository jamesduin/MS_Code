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
from sklearn import svm



class LearnRound:
    def __init__(self,testFold,rndNum, lvl,rndType):
        self.lvl_rndTime = 0
        self.rndNum = rndNum
        self.rndType = rndType
        self.testFold = testFold
        self.lvl = lvl

    def getClf(self,train_wt,clfType):
        if(clfType == 'LogReg'):
            classifier = linear_model.LogisticRegression(penalty='l2',
                                                         C=0.1,
                                                         tol=0.00001,
                                                         solver='liblinear',
                                                         class_weight={1: train_wt},
                                                         n_jobs=-1)
        elif(clfType == 'SVM'):
            classifier = svm.SVC(kernel='rbf', cache_size=8192,
                                decision_function_shape = 'ovo',
                                 class_weight={1: train_wt},
                                C=0.15,
                                 gamma=0.002)
        return classifier

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

    def createTestSet(self,test_part):
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

    def printConfMatrix(self,y_testCoarse,y_predCoarse,results):
        ###### print conf matrix,accuracy and f1_score
        confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
        acc = accuracy_score(y_testCoarse, y_predCoarse)
        f1 = f1_score(y_testCoarse, y_predCoarse)
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['conf']+['tn']+[confMatrix[0][0]] +['fp']+ [confMatrix[0][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['conf']+['fn']+[confMatrix[1][0]] +['tp']+ [confMatrix[1][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['acc']+[acc] +['f1']+[f1])


    def printConfMatrixThresh(self,y_testCoarse,y_predCoarse,results,xaxislb,xaxis,yaxislb,yaxis,threshlb,thresh):
        ###### print conf matrix,accuracy and f1_score
        # confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
        # acc = accuracy_score(y_testCoarse, y_predCoarse)
        # f1 = f1_score(y_testCoarse, y_predCoarse)
        confMatrix = [[0,0],[0,0]]
        acc = 0
        f1 = 0
        results.append(['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       #+[xaxislb]+[xaxis]+[yaxislb]+[yaxis]+[threshlb]+[thresh]
                       +['co']+['tNg']+[confMatrix[0][0]] +['fPs']+ [confMatrix[0][1]])
        results.append(['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                 #+ [xaxislb] + [xaxis] + [yaxislb] + [yaxis] + [threshlb] + [thresh]
                       +['co']+['fNg']+[confMatrix[1][0]] +['tPs']+ [confMatrix[1][1]])
        results.append(['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                 + [xaxislb] + [xaxis] + [yaxislb] + [yaxis] + [threshlb] + [thresh]
                       +['ac']+['{:.3f}'.format(acc)] +['fmes']+[f1])


    def plotRocCurves(self,y_testCoarse,y_pred_score,y_sampleWeight,results):
        ###### Plot ROC and PR curves
        fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score, drop_intermediate=False)#, sample_weight=y_sampleWeight)
        roc_auc = auc(fpr, tpr, reorder=True)
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['roc']+[roc_auc])
        return fpr, tpr, threshRoc

    def plotPrCurves(self, y_testCoarse, y_pred_score, y_sampleWeight, results):
        ##### Plog pr_curve
        precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_pred_score)#, sample_weight=y_sampleWeight)
        pr_auc = auc(recall, precision)
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]
                       +['lvl']+[self.lvl]+['pr']+ [pr_auc])
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['rndTime'] + [str(round(time.perf_counter() - self.lvl_rndTime, 2))])
        return precision, recall, threshPr

    def predictTestSetThreshold(self,thresh,y_pred_score):
        y_predCoarse = []
        for inst in y_pred_score:
            if (inst > thresh):
                y_predCoarse.append(1.0)
            else:
                y_predCoarse.append(0.0)
        return np.array(y_predCoarse),y_pred_score


class CoarseRound(LearnRound):
    def __init__(self,testFold,rndNum,rndType):
        LearnRound.__init__(self, testFold,rndNum, 'coarse',rndType)
        self.clf = []
        self.train_wt = 0.0

    def createTrainWtYtrain(self,y_train,results):
        ##### create train_wt and y_train for coarse
        y_trainCoarse = []
        for i in y_train:
            if i > 0:
                y_trainCoarse.append(1.0)
            else:
                y_trainCoarse.append(0.0)
        train_wt = fcnSclWeight(len(y_train) / np.sum(y_trainCoarse))
        self.train_wt = train_wt
        addPrint(results,'coarseTrainWt: {}'.format(self.train_wt))
        return y_trainCoarse


    def trainClassifier(self,X_train,y_trainCoarse,clfType):
        ##### Train classifier for coarse
        classifier = self.getClf(self.train_wt,clfType)
        self.clf = classifier.fit(X_train, y_trainCoarse)

    def predictTestSet(self,X_test):
        ##### Predict test set for coarse
        y_predCoarse = self.clf.predict(X_test)
        y_pred_score = self.clf.decision_function(X_test)
        return y_predCoarse,y_pred_score




class FineRound(LearnRound):
    def __init__(self,testFold,rndNum,rndType):
        LearnRound.__init__(self, testFold,rndNum, 'fine',rndType)
        self.classifier = dict()
        self.Fine_wt = []

    def createTrainWtYtrain(self,y_train,results):
        ##### create train_wt (y_train unmodified) for fine
        y_trainBin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
        wt = len(y_train) / np.sum(y_trainBin)
        train_wt = fcnSclWeight(wt)
        self.Fine_wt = np.array(
            [3.0, 1.0, 1.0, 1.5,
             10.0, 2.0, 3.0, 1.0]) * train_wt
        addPrint(results, 'fineTrainWt: {},{},{},{},{},{},{},{}'.format(*self.Fine_wt))
        return y_trainBin

    def trainClassifier(self,X_train,y_trainBin,clfType):
        #### train classifier for fine
        for cls in range(8):
            classif = self.getClf(self.Fine_wt[cls],clfType)
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
                y_predCoarse.append(1.)
            else:
                y_predCoarse.append(0.)
        return np.array(y_predCoarse),y_pred_score




def fcnSclWeight(input):
    #return input
    y = np.array([23.0, 7.5])
    x = np.array([20.8870, 4.977])
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m * input + b



















def confEstAdd(results,classes_all,sets,rndLvl,lvl,add):
    decFcn = []

    for cls in sorted(classes_all):
        partition = np.asarray(classes_all[cls])
        data = partition
        if (len(data) > 0):
            y_train, X_train = data[:, 0], data[:, 1:]
            y_predCoarse, scores = rndLvl.predictTestSet(X_train)
            s_tmp = np.abs(scores).reshape(scores.shape[0], 1)
            s_cls = np.ones(len(s_tmp)).reshape(len(s_tmp),1)*cls
            s_ind = np.array(range(len(s_tmp))).reshape(len(s_tmp),1)
            prefixCols = np.hstack((s_cls, s_ind))
            decFcnTmp = np.hstack((prefixCols,s_tmp))
            if decFcn == []:
                decFcn = decFcnTmp
            else:
                decFcn = np.vstack((decFcnTmp,decFcn))
    d_ind = np.array(range(len(decFcn))).reshape(len(decFcn), 1)
    decFcn = np.hstack((d_ind,decFcn))
    decFcn = decFcn[decFcn[:,3].argsort()].tolist()

    removeInd = []
    for i in range(add):
        most_cls = int(decFcn[i][1])
        most_ind = int(decFcn[i][2])
        removeInd.append([most_cls,most_ind])
        sets[most_cls].append(classes_all[most_cls][most_ind])
        # addPrint(results,[lvl]+['cls']+[most_cls]+['ind']+
        # [most_ind]+['mostUncert']+[most_uncert])
    addPrint(results, ['add'+lvl] + [add])

    removeInd = np.array(removeInd)
    removeInd = removeInd[removeInd[:,1].argsort()[::-1]]
    for i,inst in enumerate(removeInd):
        most_cls = int(removeInd[i][0])
        most_ind = int(removeInd[i][1])
        del classes_all[most_cls][most_ind]

def randAdd(classes,set,Addnum):
    data = []
    for x in sorted(classes):
        partition = np.asarray(classes[x])
        if (len(partition) > 0):
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
    np.random.shuffle(data)
    for i in range(Addnum):
        findAddInstance(classes,set,data[i].tolist())



def findAddInstance(classes, set, find_inst):
    for cls in sorted(classes):
        for index, inst in enumerate(classes[cls]):
            if (inst == find_inst):
                set[inst[0]].append(classes[cls].pop(index))
                return






def appendRndTimesFoldCnts(testFold, rndNum,lvl,results,set,start_time):
    addPrint(results,['rnd']+[rndNum] +['fold']+[testFold]+['lvl']+[lvl]
   +['rndTimeTot'] + [str(round(time.perf_counter() - start_time[-1], 2))])
    instanceCount = 0
    fold_cnt = ['rnd']+[rndNum] +['fold']+[testFold]+['lvl']+[lvl]
    for i in sorted(set):
        instanceCount += len(set[i])
        fold_cnt.append((i, len(set[i])))
    fold_cnt.append(('tot', instanceCount))
    addPrint(results,fold_cnt)
    return instanceCount


def appendSetTotal(rndNum, results, sets,name):
    instanceCount = 0
    fold_cnt = ['rnd'] + [rndNum] + [name]
    for i in sorted(sets):
        instanceCount += len(sets[i])
        fold_cnt.append((i, len(sets[i])))
    fold_cnt.append(('tot', instanceCount))
    addPrint(results, fold_cnt)


def switchClass5instance(test_part,train_part):
    for i, inst in enumerate(test_part):
        if inst[0] == 5:
            for part in train_part:
                train_part[part].append(test_part.pop(i))
                return





def printClsVsFolds(results,folds, title):
    addPrint(results,'{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(title, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    instanceCount = 0
    classCountTot = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in sorted(folds):
        classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        instanceCount += len(folds[i])
        for inst in folds[i]:
            classCountTot[int(inst[0])] += 1
            classCount[int(inst[0])] += 1
        classCount = [i] + [len(folds[i])] + classCount
        addPrint(results,'{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(*classCount))
    addPrint(results,'{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format('Total', instanceCount, *classCountTot))
    return classCount


def printClassTotals(results,classes):
    addPrint(results,'{} & {} \\\\'.format('Classes', ''))
    instanceCount = 0
    for i in sorted(classes):
        instanceCount += len(classes[i])
        addPrint(results,'{} & {} \\\\'.format(i, len(classes[i])))
    addPrint(results,'{} & {} \\\\'.format('Total', instanceCount))
    addPrint(results,'{} & {} \\\\'.format('Shape',len(classes[0][0])))

def addPrint(results,x):
    results.append(x)
    print(x)



def loadScaledPartData(loadDir):
    all_part = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for i in sorted(all_part):
        with open(loadDir + str(i)) as f:
            for line in f:
                nums = line.split()
                nums = list(map(float, nums))
                all_part[i].append(nums)
        np.random.shuffle(all_part[i])
    return all_part




