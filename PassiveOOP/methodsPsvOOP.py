import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve,f1_score
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
import numpy as np
import time


class LearnRound:
    def __init__(self, lvl, folds, fName):
        self.fName = fName
        self.lvl = lvl
        self.testFold = 0.0
        self.folds = folds
        self.results = []
        self.y_train = []
        self.X_train = []
        self.min_max_scaler = []
        self.y_testCoarse = []
        self.X_test = []
        self.y_sampleWeight = []

        self.f = open('results/_' + self.fName + '.txt', 'w')

    def createTrainSet(self):
        ##### create train set
        print(self.lvl + " fold" + str(self.testFold))
        self.f.write('{} fold {}\n'.format(self.lvl, self.testFold))
        data = []
        partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        partition_list.remove(self.testFold)
        for x in partition_list:
            partition = np.asarray(self.folds[x])
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
        self.y_train, X_trainPreScale = data[:, 0], data[:, 1:data.shape[1]]
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.X_train = self.min_max_scaler.fit_transform(X_trainPreScale)

    def createTestSet(self):
        ##### create test set and sample weight
        data_test = np.asarray(self.folds[self.testFold])
        y_test, X_testPreScale = data_test[:, 0], data_test[:, 1:data_test.shape[1]]
        self.X_test = self.min_max_scaler.transform(X_testPreScale)
        self.y_testCoarse = []
        self.y_sampleWeight = []
        for inst in y_test:
            if inst > 0:
                self.y_testCoarse.append(1.0)
            else:
                self.y_testCoarse.append(0.0)

        test_wt = len(self.y_testCoarse) / np.sum(self.y_testCoarse)
        for inst in self.y_testCoarse:
            if inst > 0:
                self.y_sampleWeight.append(test_wt)
            else:
                self.y_sampleWeight.append(1.0)

    def printConfMatrix(self):
        ###### print conf matrix,accuracy and f1_score
        confMatrix = confusion_matrix(self.y_testCoarse, self.y_predCoarse)
        print(confMatrix)
        self.f.write(
            '[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0], confMatrix[0][1], confMatrix[1][0],
                                                  confMatrix[1][1]))
        print(accuracy_score(self.y_testCoarse, self.y_predCoarse))
        self.f.write('acc_score: {:.3f}\n'.format(accuracy_score(self.y_testCoarse, self.y_predCoarse)))
        print(f1_score(self.y_testCoarse, self.y_predCoarse))
        self.f.write('f1_score: {:.3f}\n'.format(f1_score(self.y_testCoarse, self.y_predCoarse)))
        self.results.append([' ']+[confMatrix[0][0]] + [confMatrix[0][1]]  + [' ']+ [np.round(self.wt,3)])
        self.results.append([' '] + [confMatrix[1][0]] + [confMatrix[1][1]] + [' '] + [np.round(self.train_wt,3)])


    def plotRocPrCurves(self):
        ###### Plot ROC and PR curves
        fpr, tpr, threshRoc = roc_curve(self.y_testCoarse, self.y_pred_score, sample_weight=self.y_sampleWeight)
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
        plt.savefig(self.lvl + '_results/' + self.fName + '_ROC_' + str(self.testFold) + '.png')

        ##### Plot pr_curve
        precision, recall, threshPr = precision_recall_curve(self.y_testCoarse, self.y_pred_score, sample_weight=self.y_sampleWeight)
        pr_auc = auc(recall, precision)
        plt.clf()
        plt.plot(recall, precision, color='blue', linewidth=4, linestyle=':',
                 label='Precision-recall curve (area = {0:0.3f})'.format(pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="lower right")
        plt.savefig(self.lvl + '_results/' + self.fName + '_PR_' + str(self.testFold) + '.png')
        self.results.append([str(self.testFold)]  + [pr_auc] + [roc_auc]+[accuracy_score(self.y_testCoarse, self.y_predCoarse)]+[f1_score(self.y_testCoarse, self.y_predCoarse)])

    # def saveResults(self,start_time):
    #     ###### Save results to a file
    #     self.f.write(self.lvl + '\n')
    #     self.f.write('{0:5}{1:7}{2:7}\n'.format('fold', 'roc', 'pr'))
    #     roc_Sum = 0.0
    #     pr_Sum = 0.0
    #     for result in self.results:
    #         self.f.write('{0:<5}{1:<7.3f}{2:<7.3f}\n'.format(*result))
    #         roc_Sum += result[1]
    #         pr_Sum += result[2]
    #     #self.f.write('{0:},{1:.3f},{2:.3f} \n'.format('avg', (roc_Sum / len(self.results)), (pr_Sum / len(self.results))))
    #     #print('{0:},{1:.3f},{2:.3f}'.format('avg', (roc_Sum / len(self.results)), (pr_Sum / len(self.results))))
    #
    #     print('{} sec'.format(round(time.perf_counter() - start_time, 2)))
    #     self.f.write('{} sec'.format(round(time.perf_counter() - start_time, 2)))
    #     self.f.close()









class CoarseRound(LearnRound):
    def __init__(self,coarse_folds,fName):
        LearnRound.__init__(self,'coarse',coarse_folds,fName)
        self.train_wt = 0.0
        self.wt = 0.0
        self.y_trainCoarse = []
        self.clf = []
        self.y_predCoarse = []
        self.y_pred_score = []

    def createTrainWtYtrain(self):
        ##### create train_wt and y_train for coarse
        self.y_trainCoarse = []
        for i in self.y_train:
            if i > 0:
                self.y_trainCoarse.append(1.)
            else:
                self.y_trainCoarse.append(0.)
        self.wt =len(self.y_train) / np.sum(self.y_trainCoarse)
        self.train_wt = fcnSclWeight(self.wt)
        #self.train_wt = self.wt

    def trainClassifier(self):
        ##### train classifier for coarse
        classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                     fit_intercept=False, intercept_scaling=1,
                                                     class_weight={1: self.train_wt},
                                                     solver='liblinear',
                                                     max_iter=1000, n_jobs=-1)
        self.clf = classifier.fit(self.X_train, self.y_trainCoarse)
        joblib.dump(self.clf, self.lvl + '_models/' + self.fName + '_' + str(self.testFold) + '.pkl')

    def predictTestSet(self):
        ##### predict test set for coarse
        self.y_predCoarse = self.clf.predict(self.X_test)
        self.y_pred_score = self.clf.decision_function(self.X_test)










class FineRound(LearnRound):
    def __init__(self,fine_folds,fName):
        LearnRound.__init__(self,'fine',fine_folds,fName)
        self.train_wt = 0.0
        self.wt = 0.0
        self.Fine_wt = []
        self.y_trainBin = []
        self.y_trainCoarse = []
        self.classifier = dict()
        self.y_predCoarse = []
        self.y_pred_score = []

    def createTrainWtYtrain(self):
        ##### create train_wt (y_train unmodified) for fine
        self.y_trainBin = label_binarize(self.y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
        self.wt = len(self.y_train) / np.sum(self.y_trainBin)
        self.train_wt = fcnSclWeight(self.wt)
        self.Fine_wt = np.array([1,0.5,0.9,0.75    ,4,0.9,2,1])*self.train_wt



    def trainClassifier(self):
        #### train classifier for fine
        for cls in range(8):
            classif = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                      fit_intercept=False, intercept_scaling=1,
                                                      class_weight={1: self.Fine_wt[cls]},
                                                      solver='liblinear',
                                                      max_iter=1000, n_jobs=-1)
            clf = classif.fit(self.X_train, self.y_trainBin[:, cls])
            joblib.dump(clf, self.lvl + '_models/' + self.fName + '_' + str(self.testFold) + '_' + str(cls + 1) + '.pkl')
            self.classifier[cls] = clf

    def predictTestSet(self):
        ##### predict test set for fine
        y_fine_score = []
        for cls in range(8):
            scores = self.classifier[cls].decision_function(self.X_test)
            scores = scores.reshape(scores.shape[0], 1)
            if y_fine_score == []:
                y_fine_score = scores
            else:
                y_fine_score = np.hstack((y_fine_score, scores))
        self.y_pred_score = np.amax(y_fine_score, axis=1)
        self.y_predCoarse = []
        for inst in self.y_pred_score:
            if (inst > 0.0):
                self.y_predCoarse.append(1.)
            else:
                self.y_predCoarse.append(0.)







def fcnSclWeight(input):
    #return input
    y = np.array([20.0, 6.5])
    x = np.array([20.8870, 4.977])
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m * input + b
    #return 0.685331066*x+6.5884



def printClsVsFolds(folds, title):
    # stdout.write(
    #     '{:<14}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}\n'.format(title, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    print('{:<14}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(title, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    instanceCount = 0
    classCountTot = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in sorted(folds):
        classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        instanceCount += len(folds[i])
        for inst in folds[i]:
            classCountTot[int(inst[0])] += 1
            classCount[int(inst[0])] += 1
        classCount = [i] + [len(folds[i])] + classCount
        # stdout.write('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}\n'.format(*classCount))
        print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))
    # stdout.write(
    #     '{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}\n'.format('Total', instanceCount, *classCountTot))
    print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total', instanceCount, *classCountTot))


def printClassTotals(classes):
    # stdout.write('{0:<10}{1:<10}\n'.format('Classes', ''))
    print('{0:<10}{1:<10}'.format('Classes', ''))
    instanceCount = 0
    for i in sorted(classes):
        instanceCount += len(classes[i])
        # stdout.write('{0:<10}{1:<10}\n'.format(i, len(classes[i])))
        print('{0:<10}{1:<10}'.format(i, len(classes[i])))
    # stdout.write('{0:<10}{1:<10}\n'.format('Total', instanceCount))
    print('{0:<10}{1:<10}\n'.format('Total', instanceCount))
    # stdout.write('Shape: {0:<10}\n'.format(len(classes[0][0])))
    print('Shape: {0:<10}\n'.format(len(classes[0][0])))







