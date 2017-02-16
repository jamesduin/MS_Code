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

class LearnRound:
    def __init__(self,rndNum,FFR,lvl):
        self.lvl_rndTime = 0
        self.rndNum = rndNum
        self.FFR = FFR
        self.lvl = lvl

    def createSetParts(self,set):
        folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
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
            for i in sorted(folds):
                np.random.shuffle(folds[i])
        return folds

    def createTrainSet(self, setParts, fld):
        ##### Create train set for coarse
        data = []
        partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        partition_list.remove(fld)
        for x in partition_list:
            partition = np.asarray(setParts[x])
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
        y_train, X_train = data[:, 0], data[:, 1:]
        return y_train, X_train

    def createTestSet(self, setParts):
        ##### Create test set for coarse
        data_test = np.asarray(setParts)
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

    def printConfMatrix(self,y_testCoarse,y_predCoarse,results,fld):
        ###### print conf matrix,accuracy and f1_score
        confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
        acc = accuracy_score(y_testCoarse, y_predCoarse)
        f1 = f1_score(y_testCoarse, y_predCoarse)
        addPrint(results,['rnd']+[self.rndNum]+['FFR']+[self.FFR]+['lvl']+[self.lvl]+['fld']+[fld]
                       +['conf']+[confMatrix[0][0]] + [confMatrix[0][1]])
        addPrint(results,['rnd']+[self.rndNum]+['FFR']+[self.FFR]+['lvl']+[self.lvl]+['fld']+[fld]
                       +['conf'] +[confMatrix[1][0]] + [confMatrix[1][1]])
        addPrint(results,['rnd']+[self.rndNum]+['FFR']+[self.FFR]+['lvl']+[self.lvl]+['fld']+[fld]
                       +['acc']+[acc] +['f1']+[f1])


    def plotRocPrCurves(self,y_testCoarse,y_pred_score,y_sampleWeight,results,fld):
        ###### Plot ROC and PR curves
        fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score, sample_weight=y_sampleWeight)
        roc_auc = auc(fpr, tpr, reorder=True)
        if (self.rndNum % 50 == 0):
            plt.figure()
            plt.plot(fpr, tpr,
                     label='ROC curve (area = {0:0.3f})'.format(roc_auc),
                     color='red', linestyle=':', linewidth=4)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.savefig(self.lvl + '_results/FFR_'+ self.FFR+'_'+ str(self.rndNum) + '_' + self.lvl +'_fld_'+str(fld)+ '_ROC.png')
            plt.clf()
            plt.close()

        ##### Plog pr_curve
        precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_pred_score, sample_weight=y_sampleWeight)
        pr_auc = auc(recall, precision)
        if (self.rndNum % 50 == 0):
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2, linestyle=':',
                     label='Precision-recall curve (area = {0:0.3f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall')
            plt.legend(loc="lower right")
            plt.savefig(self.lvl + '_results/FFR_'+ self.FFR+'_'+ str(self.rndNum) + '_' + self.lvl +'_fld_'+str(fld)+ '_PR.png')
            plt.clf()
            plt.close()
        addPrint(results,['rnd']+[self.rndNum]+['FFR']+[self.FFR]+['lvl']+[self.lvl]+['fld']+[fld]
                       +['pr']+ [pr_auc]+ ['roc']+[roc_auc])
        addPrint(results,['rnd']+[self.rndNum]+['FFR']+[self.FFR]+['lvl']+[self.lvl]+['fld']+[fld]
                       +['rndTime'] + [str(round(time.perf_counter() - self.lvl_rndTime, 2))])

class CoarseRound(LearnRound):
    def __init__(self,rndNum,FFR):
        LearnRound.__init__(self,rndNum, FFR,'coarse')
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

    def trainClassifier(self,X_train,y_trainCoarse,fld):
        ##### Train classifier for coarse
        classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                     fit_intercept=False, intercept_scaling=1,
                                                     class_weight={1: self.train_wt},
                                                     solver='liblinear',
                                                     max_iter=1000, n_jobs=-1)
        self.clf = classifier.fit(X_train, y_trainCoarse)
        if (self.rndNum % 50 == 0):
            joblib.dump(self.clf, self.lvl + '_models/FFR_'+ self.FFR+'_'+ str(self.rndNum) + '_' + self.lvl +'_fld_'+str(fld)+ '.pkl')

    def predictTestSet(self,X_test):
        ##### Predict test set for coarse
        y_predCoarse = self.clf.predict(X_test)
        y_pred_score = self.clf.decision_function(X_test)
        return y_predCoarse,y_pred_score

class FineRound(LearnRound):
    def __init__(self,rndNum,FFR):
        LearnRound.__init__(self,rndNum, FFR,'fine')
        self.classifier = dict()
        self.Fine_wt = []

    def createTrainWtYtrain(self,y_train):
        ##### create train_wt (y_train unmodified) for fine
        y_trainBin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
        wt = len(y_train) / np.sum(y_trainBin)
        train_wt = fcnSclWeight(wt)
        self.Fine_wt = np.array(
            [0.8695652173913044, 0.4347826086956522, 0.782608695652174, 0.6521739130434783, 3.4782608695652177,
             0.782608695652174, 1.7391304347826089, 0.8695652173913044]) * train_wt
        return y_trainBin

    def trainClassifier(self,X_train,y_trainBin,fld):
        #### train classifier for fine
        for cls in range(8):
            classif = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                      fit_intercept=False, intercept_scaling=1,
                                                      class_weight={1: self.Fine_wt[cls]},
                                                      solver='liblinear',
                                                      max_iter=1000, n_jobs=-1)
            clf = classif.fit(X_train, y_trainBin[:, cls])
            if(self.rndNum % 50 == 0):
                joblib.dump(clf, self.lvl + '_models/FFR_'+ self.FFR+'_'+ str(self.rndNum) + '_' + self.lvl +'_fld_'+str(fld)+'_' + str(cls + 1) + '.pkl')
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
        return y_predCoarse,y_pred_score




def fcnSclWeight(input):
    #return input
    #y = np.array([20.0, 6.5])
    y = np.array([23.0, 7.475])
    x = np.array([20.8870, 4.977])
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m * input + b


























def appendRndTimesFoldCnts(FFR, rndNum,lvl,results,set,start_time):
    addPrint(results,['rnd']+[rndNum] +['FFR']+[FFR]+['lvl']+[lvl]
                   +['rndTimeTot'] + [str(round(time.perf_counter() - start_time[-1], 2))])
    instanceCount = 0
    fold_cnt = ['rnd']+[rndNum] +['FFR']+[FFR]+['lvl']+[lvl]
    for i in sorted(set):
        instanceCount += len(set[i])
        fold_cnt.append((i, len(set[i])))
    fold_cnt.append(('tot', instanceCount))
    addPrint(results,fold_cnt)
    return instanceCount




def confEstAdd(classes_all,set,rnds,add):
    decFcn = dict()
    for lvl in ['coarse', 'fine']:
        decFcn[lvl] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

    for i in sorted(classes_all):
        partition = np.asarray(classes_all[i])
        data = partition
        if (len(data) > 0):
            y_train, X_train = data[:, 0], data[:, 1:]
            for lvl in ['coarse', 'fine']:
                for fold in range(1, 11):
                    y_predCoarse, scores = rnds[lvl].predictTestSet(X_train)
                    scores = scores.reshape(scores.shape[0], 1)
                    if decFcn[lvl][i] == []:
                        decFcn[lvl][i] = scores
                    else:
                        np.hstack((decFcn[lvl][i], scores))
                decFcn[lvl][i] = decFcn[lvl][i].tolist()

    for i in range(add['coarse']):
        most_uncert = 100
        most_cls = 0
        most_ind = 0
        for cls in sorted(decFcn['coarse']):
            for index, inst in enumerate(decFcn['coarse'][cls]):
                max_est = 0.0
                for eachClassEst in inst:
                    est = np.absolute(eachClassEst)
                    if (max_est < est):
                        max_est = est
                if(max_est<most_uncert):
                    most_cls = cls
                    most_ind = index
                    most_uncert = max_est
        set['coarse'][most_cls].append(classes_all[most_cls].pop(most_ind))
        del decFcn['coarse'][most_cls][most_ind]
        del decFcn['fine'][most_cls][most_ind]

    for i in range(add['fine']):
        most_uncert = 100
        most_cls = 0
        most_ind = 0
        for cls in sorted(decFcn['fine']):
            for index, inst in enumerate(decFcn['fine'][cls]):
                max_est = 0.0
                for eachClassEst in inst:
                    est = np.absolute(eachClassEst)
                    if (max_est < est):
                        max_est = est
                if (max_est < most_uncert):
                    most_cls = cls
                    most_ind = index
                    most_uncert = max_est
        set['coarse'][most_cls].append(classes_all[most_cls][most_ind])
        set['fine'][most_cls].append(classes_all[most_cls].pop(most_ind))
        del decFcn['coarse'][most_cls][most_ind]
        del decFcn['fine'][most_cls][most_ind]




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
            for line in f:
                nums = line.split()
                nums = list(map(float, nums))
                all_part[i].append(nums)
        np.random.shuffle(all_part[i])
    return all_part




