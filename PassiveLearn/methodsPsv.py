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
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import time
from sklearn import svm
import pprint as pp


def fcnSclWeight(input):
    #return input
    #y = np.array([20.0, 6.5])
    #y = np.array([23.0, 7.475])
    y = np.array([23.0, 7.5])
    x = np.array([20.8870, 4.977])
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m * input + b
    #return input




class LearnRound:
    def __init__(self,testFold,rndNum, lvl,Psv):
        self.lvl_rndTime = 0
        self.rndNum = rndNum
        self.Psv = Psv
        self.testFold = testFold
        self.lvl = lvl


    def getClf(self,train_wt):
        classifier = linear_model.LogisticRegression(penalty='l2',
                                                     C=0.1,
                                                     tol=0.00001,
                                                     solver='liblinear',
                                                     class_weight={1: train_wt},
                                                     n_jobs=-1)

        # classifier = svm.SVC(kernel='rbf', cache_size=8192,
        #                     decision_function_shape = 'ovo',
        #                      class_weight={1: train_wt},
        #                     C=0.15,
        #                      gamma=0.002)
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
        confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
        acc = accuracy_score(y_testCoarse, y_predCoarse)
        f1 = f1_score(y_testCoarse, y_predCoarse)
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                       #+[xaxislb]+[xaxis]+[yaxislb]+[yaxis]+[threshlb]+[thresh]
                       +['co']+['tNg']+[confMatrix[0][0]] +['fPs']+ [confMatrix[0][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                 #+ [xaxislb] + [xaxis] + [yaxislb] + [yaxis] + [threshlb] + [thresh]
                       +['co']+['fNg']+[confMatrix[1][0]] +['tPs']+ [confMatrix[1][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+[self.lvl]
                 + [xaxislb] + [xaxis] + [yaxislb] + [yaxis] + [threshlb] + [thresh]
                       +['ac']+['{:.3f}'.format(acc)] +['fmes']+[f1])


    def plotRocCurves(self,y_testCoarse,y_pred_score,y_sampleWeight,results):
        ###### Plot ROC and PR curves
        fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score  ,drop_intermediate=False)#, sample_weight=y_sampleWeight)
        roc_auc = auc(fpr, tpr, reorder=True)
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
        plt.savefig(self.lvl + '_results/'+ self.Psv+'_'+ str(self.rndNum) + '_' + str(self.testFold) +'_' + self.lvl + '_ROC.png')
        plt.clf()
        plt.close()
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['roc']+[roc_auc])
        return fpr, tpr, threshRoc

    def plotPrCurves(self, y_testCoarse, y_pred_score, y_sampleWeight, results):
        ##### Plog pr_curve
        precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_pred_score)#, sample_weight=y_sampleWeight)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, linestyle=':',
                 label='Precision-recall curve (area = {0:0.3f})'.format(pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="lower right")
        plt.savefig(self.lvl + '_results/'+ self.Psv+'_'+ str(self.rndNum) +  '_' + str(self.testFold) +'_' + self.lvl + '_PR.png')
        plt.clf()
        plt.close()
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]
                       +['lvl']+[self.lvl]+['pr']+ [pr_auc])
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+[self.lvl]
                       +['rndTime'] + [str(round(time.perf_counter() - self.lvl_rndTime, 2))])
        return precision, recall, threshPr



class CoarseRound(LearnRound):
    def __init__(self,testFold,rndNum,Psv):
        LearnRound.__init__(self, testFold,rndNum, 'coarse',Psv)
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

    def trainClassifier(self,X_train,y_trainCoarse):
        ##### Train classifier for coarse
        classifier = self.getClf(self.train_wt)
        self.clf = classifier.fit(X_train, y_trainCoarse)
        #joblib.dump(self.clf, self.lvl + '_models/'+ self.Psv+'_'+ str(self.rndNum) + '_' + self.lvl + '.pkl')
        #self.clf = joblib.load(self.lvl + '_models/'+ self.Psv+'_'+ str(self.rndNum) + '_' + self.lvl + '.pkl')

    def predictTestSet(self,X_test):
        ##### Predict test set for coarse
        y_predCoarse = self.clf.predict(X_test)
        y_pred_score = self.clf.decision_function(X_test)
        return y_predCoarse,y_pred_score

    def predictTestSetThreshold(self,thresh,y_pred_score):
        y_predCoarse = []
        y_predCoarse = []
        for inst in y_pred_score:
            if (inst > thresh):
                y_predCoarse.append(1.0)
            else:
                y_predCoarse.append(0.0)

        return np.array(y_predCoarse),y_pred_score



class FineRound(LearnRound):
    def __init__(self,testFold,rndNum,Psv):
        LearnRound.__init__(self, testFold,rndNum, 'fine',Psv)
        self.classifier = dict()
        self.Fine_wt = []

    def createTrainWtYtrain(self,y_train,results):
        ##### create train_wt (y_train unmodified) for fine
        y_trainBin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
        wt = len(y_train) / np.sum(y_trainBin)
        train_wt = fcnSclWeight(wt)
        # self.Fine_wt = np.array(
        #     [0.8695652173913044, 0.4347826086956522, 0.782608695652174, 0.6521739130434783,
        #      3.4782608695652177, 0.782608695652174, 1.7391304347826089, 0.8695652173913044]) * train_wt
        self.Fine_wt = np.array(
            [3.0, 1.0, 1.0, 1.5,
             10.0, 2.0, 3.0, 1.0]) * train_wt
        addPrint(results, 'fineTrainWt: {},{},{},{},{},{},{},{}'.format(*self.Fine_wt))
        return y_trainBin

    def trainClassifier(self,X_train,y_trainBin):
        #### train classifier for fine
        for cls in range(8):
            classif = self.getClf(self.Fine_wt[cls])
            clf = classif.fit(X_train, y_trainBin[:, cls])
            #joblib.dump(clf, self.lvl + '_models/'+ self.Psv+'_'+ str(self.rndNum) + '_' + str(self.lvl) + '_' + str(cls + 1) + '.pkl')
            # clf = joblib.load(self.lvl + '_models/'+ self.Psv+'_'+ str(self.rndNum) + '_' + str(self.lvl) + '_' + str(cls + 1) + '.pkl')
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

    def predictTestSetFineCls(self,X_test,y_test,fineCls,label,results):
        ##### predict test set for fine
        y_predFine = self.classifier[fineCls-1].predict(X_test)
        y_pred_score = self.classifier[fineCls-1].decision_function(X_test)
        y_testFine = label_binarize(y_test, classes=[fineCls])
        ###### print conf matrix,accuracy and f1_score
        confMatrix = confusion_matrix(y_testFine, y_predFine)
        acc = accuracy_score(y_testFine, y_predFine)
        f1 = f1_score(y_testFine, y_predFine)
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+
                 [label]+['conf']+['tn']+[confMatrix[0][0]] +['fp']+ [confMatrix[0][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+
                 [label]+['conf']+['fn']+[confMatrix[1][0]] +['tp']+ [confMatrix[1][1]])
        addPrint(results,['rnd']+[self.rndNum] +['fold']+[self.testFold]+['lvl']+
                 [label]+['acc']+[acc] +['f1']+[f1])

        y_sampleWeight = []
        test_wt = len(y_testFine) / np.sum(y_testFine)
        for inst in y_testFine:
            if inst > 0:
                y_sampleWeight.append(test_wt)
            else:
                y_sampleWeight.append(1.0)
        fpr, tpr, threshRoc = roc_curve(y_testFine, y_pred_score, sample_weight=y_sampleWeight)
        roc_auc = auc(fpr, tpr, reorder=True)
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+
                 [label]+['roc']+ [roc_auc])

        precision, recall, threshPr = precision_recall_curve(y_testFine, y_pred_score, sample_weight=y_sampleWeight)
        pr_auc = auc(recall, precision)
        addPrint(results,['rnd']+[self.rndNum]+['fold']+[self.testFold]+['lvl']+
                 [label]+['pr']+ [pr_auc])



    def predictTestSetThreshold(self,thresh,y_pred_score):
        ##### predict test set for fine
        y_predCoarse = []
        for inst in y_pred_score:
            if (inst > thresh):
                y_predCoarse.append(1.0)
            else:
                y_predCoarse.append(0.0)
        return np.array(y_predCoarse),y_pred_score






























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
    return y_testCoarse, y_sampleWeight, X_test, y_test


def predictCombined(results,y_pred_score,y_testCoarse,y_sampleWeight,rndNum,testFold,Psv):
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
             + ['conf'] +['tn']+ [confMatrix[0][0]] +['fp']+ [confMatrix[0][1]])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['conf'] +['fn']+ [confMatrix[1][0]] +['tp']+ [confMatrix[1][1]])
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
             + ['combConfMat'] + ['tn_inf'] +combConfMat['tn'])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['fp_inf'] +combConfMat['fp'])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['fn_inf'] +combConfMat['fn'])
    addPrint(results, ['rnd'] + [rndNum] + ['fold'] + [testFold] + ['combPred']
             + ['combConfMat'] + ['tp_inf'] +combConfMat['tp'])

    ###### Plot ROC and PR curves
    fpr, tpr, threshRoc = roc_curve(y_testCoarse, combPredScore, sample_weight=y_sampleWeight)
    roc_auc = auc(fpr, tpr, reorder=True)
    # plt.figure()
    # plt.plot(fpr, tpr,
    #          label='ROC curve (area = {0:0.3f})'.format(roc_auc),
    #          color='red', linestyle=':', linewidth=4)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('comb_results/'+ str(Psv)+'_'+ str(rndNum) + '_' + str(testFold) +'_comb_ROC.png')
    # plt.clf()
    # plt.close()

    ##### Plog pr_curve
    precision, recall, threshPr = precision_recall_curve(y_testCoarse, combPredScore, sample_weight=y_sampleWeight)
    pr_auc = auc(recall, precision)

    # plt.figure()
    # plt.plot(recall, precision, color='blue', lw=2, linestyle=':',
    #          label='Precision-recall curve (area = {0:0.3f})'.format(pr_auc))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall')
    # plt.legend(loc="lower right")
    # plt.savefig('comb_results/'+ str(Psv)+'_'+ str(rndNum) + '_' + str(testFold) +'_comb_PR.png')
    # plt.clf()
    # plt.close()
    addPrint(results,['rnd']+[rndNum] +['fold']+[testFold]
                   +['comb']+['pr']+ [pr_auc]+ ['roc']+[roc_auc])




def appendSetTotal(rndNum, results, sets,name,testFold):
    instanceCount = 0
    fold_cnt = ['rnd']+[rndNum] +['fold']+[testFold]+ [name]
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
        for i in sorted(sets):
            instanceCount[lvl] += len(sets[i])
            fold_cnt.append((i, len(sets[i])))
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



def printDataInstance(instance, file):
    for dimension in instance[:-1]:
        file.write(str(dimension) + " ")
    file.write(str(instance[-1]))
    file.write("\n")
    return




def logErrors(clfType, lvl, testFold, y_predCoarse,y_testCoarse,y_test,X_test):
    ###### log the errors
    err_file = open('jaccard/'+clfType+'_'+lvl+'_'+str(testFold)+'.txt', 'w')
    for i,pred in enumerate(y_predCoarse[lvl]):
        if(y_predCoarse[lvl][i] != y_testCoarse[i]):
            printDataInstance(np.array([y_test[i]]+list(X_test[i])), err_file)
    err_file.close()