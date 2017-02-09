import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve,f1_score
from sklearn import preprocessing
import numpy as np
import time


class LearnRound:

    def __init__(self, lvl, testFold, folds, fName):
        self.fName = fName
        self.lvl = lvl
        self.testFold = testFold
        self.folds = folds
        self.results = []
        self.f = open('results/_' + self.fName + '.txt', 'w')
        print(self.lvl + " fold" + str(self.testFold))
        self.f.write('{} fold {}\n'.format(self.lvl, self.testFold))

    def createTrainSet(self):
        data = []
        partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        partition_list.remove(self.testFold)
        for x in partition_list:
            partition = np.asarray(self.folds[x])
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
        y_train, X_trainPreScale = data[:, 0], data[:, 1:data.shape[1]]
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_trainPreScale)
        return y_train,X_train,min_max_scaler

    def createTestSet(self, min_max_scaler, train_wt):
        data_test = np.asarray(self.folds[self.testFold])
        y_test, X_testPreScale = data_test[:, 0], data_test[:, 1:data_test.shape[1]]
        X_test = min_max_scaler.transform(X_testPreScale)
        y_testCoarse = []
        for inst in y_test:
            if inst > 0:
                y_testCoarse.append(1.0)
            else:
                y_testCoarse.append(0.0)
        y_sampleWeight = []
        for inst in y_testCoarse:
            if inst > 0:
                y_sampleWeight.append(train_wt)
            else:
                y_sampleWeight.append(1.0)
        return y_testCoarse, X_test, y_sampleWeight



    def printConfMatrix(self, y_testCoarse, y_predCoarse):
        print("cumulative")
        self.f.write("cumulative\n")
        confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
        print(confMatrix)
        self.f.write(
            '[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0], confMatrix[0][1], confMatrix[1][0],
                                                  confMatrix[1][1]))
        print(accuracy_score(y_testCoarse, y_predCoarse))
        self.f.write('acc_score: {:.3f}\n'.format(accuracy_score(y_testCoarse, y_predCoarse)))
        print(f1_score(y_testCoarse, y_predCoarse))
        self.f.write('f1_score: {:.3f}\n'.format(f1_score(y_testCoarse, y_predCoarse)))

    def plotRocPrCurves(self, y_testCoarse, y_pred_score, y_sampleWeight):
        ##### Plot roc_auc
        fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score, sample_weight=y_sampleWeight)
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

        ##### Plog pr_curve
        precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_pred_score, sample_weight=y_sampleWeight)
        average_precision = auc(recall, precision)
        plt.clf()
        plt.plot(recall, precision, color='blue', lw=2, linestyle=':',
                 label='Precision-recall curve (area = {0:0.3f})'.format(average_precision))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="lower right")
        plt.savefig(self.lvl + '_results/' + self.fName + '_PR_' + str(self.testFold) + '.png')
        self.results.append([str(self.testFold)] + [roc_auc] + [average_precision])

    def saveResults(self,start_time):
        ###### Save results to a file
        self.f.write(self.lvl + '\n')
        self.f.write('{0:5}{1:7}{2:7}\n'.format('fold', 'roc', 'pr'))
        roc_Sum = 0.0
        pr_Sum = 0.0
        for result in self.results:
            self.f.write('{0:<5}{1:<7.3f}{2:<7.3f}\n'.format(*result))
            roc_Sum += result[1]
            pr_Sum += result[2]
        self.f.write('{0:},{1:.3f},{2:.3f} \n'.format('avg', (roc_Sum / len(self.results)), (pr_Sum / len(self.results))))
        print('{0:},{1:.3f},{2:.3f} \n'.format('avg', (roc_Sum / len(self.results)), (pr_Sum / len(self.results))))

        print('{} sec'.format(round(time.perf_counter() - start_time, 2)))
        self.f.write('{} sec'.format(round(time.perf_counter() - start_time, 2)))
        self.f.close()


def fcnSclWeight(x):
    return 0.685331066*x+6.5884







