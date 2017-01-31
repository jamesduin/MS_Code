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
from sklearn import linear_model
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
        for i in sorted(folds):
            np.random.shuffle(folds[i])


def iterateFoldsCoarse(rndNum, coarse_folds,rnd_results_coarse):
    fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # fold_list = [1]
    rnd_results_coarse.append(['rnd','fold', 'roc_auc', 'pr_auc'])
    for testFold in fold_list:
        print("rnd"+str(rndNum)+"_coarse fold" + str(testFold))
        ##### Create train set for coarse
        data = []
        partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        partition_list.remove(testFold)
        for x in partition_list:
            partition = np.asarray(coarse_folds[x])
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

        ##### Create test set for coarse
        data_test = np.asarray(coarse_folds[testFold])
        y_test, X_test = data_test[:, 0], data_test[:, 1:data_test.shape[1]];

        y_testCoarse = []
        for i in y_test:
            if i > 0:
                y_testCoarse.append(1.)
            else:
                y_testCoarse.append(0.)

        ##### Train classifier for coarse
        classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                     fit_intercept=False, intercept_scaling=1, class_weight={1: 30},
                                                     solver='liblinear',
                                                     max_iter=1000, n_jobs=-1)
        clf = classifier.fit(X_train, y_trainCoarse)
        joblib.dump(clf, 'coarse_models/rnd'+str(rndNum)+'_coarse_fold_' + str(testFold) + '.pkl')
        # clf = joblib.load('coarse_models/rnd'+str(rndNum)+'_coarse_fold_' + str(testFold) + '.pkl')
        y_pred = clf.predict(X_test);

        ##### Predict test set for coarse
        y_predCoarse = []
        for i in y_pred:
            if i > 0:
                y_predCoarse.append(1.)
            else:
                y_predCoarse.append(i)

        y_score = clf.decision_function(X_test)


        ###### Print this folds roc_auc for coarse
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[1], tpr[1], thresholds = roc_curve(y_testCoarse, y_score)
        roc_auc[1] = auc(fpr[1], tpr[1])
        # Plot of a ROC curve
        plt.figure()
        plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.5f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('results/rnd'+str(rndNum)+'_coarse_ROC_' + str(testFold) + '.png')
        plt.clf()
        #plt.close()

        ###### Print this folds pr_auc for coarse
        precision = dict()
        recall = dict()
        average_precision = dict()
        precision[1], recall[1], _ = precision_recall_curve(y_testCoarse, y_score)
        average_precision[1] = average_precision_score(y_testCoarse, y_score)
        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[1], precision[1], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall: AUC={0:0.5f}'.format(average_precision[1]))
        plt.legend(loc="lower left")
        # plt.show()
        plt.savefig('results/rnd'+str(rndNum)+'_coarse_PR_' + str(testFold) + '.png')
        plt.clf()
        plt.close()
        rnd_results_coarse.append([rndNum]+[testFold] + [roc_auc[1]] + [average_precision[1]])











def iterateFoldsFine(rndNum,fine_folds,rnd_results_fine):
    fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # fold_list = [1]
    rnd_results_fine.append(['rnd','fold', 'roc_auc', 'pr_auc'])
    for testFold in fold_list:
        print("rnd"+str(rndNum)+"_fine fold" + str(testFold))
        ##### Create train set for fine
        data = []
        partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        partition_list.remove(testFold)
        for x in partition_list:
            partition = np.asarray(fine_folds[x])
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
        y_train, X_train = data[:, 0], data[:, 1:data.shape[1]]

        y_trainSets = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
        for cls in sorted(y_trainSets):
            for i in y_train:
                if (i == cls):
                    y_trainSets[cls].append(1.)
                else:
                    y_trainSets[cls].append(0.)


        for cls in sorted(y_trainSets):
            y_train = y_trainSets[cls]

            ##### Train classifier for fine
            classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                         fit_intercept=False, intercept_scaling=1, class_weight={1: 30},
                                                         solver='liblinear',
                                                         max_iter=1000, n_jobs=-1)

            clf = classifier.fit(X_train, y_train)
            joblib.dump(clf, 'fine_models/rnd'+str(rndNum)+'fine_fold_' + str(testFold) + '_' + str(cls) + '.pkl')
            # clf = joblib.load('fine_models/rnd'+str(rndNum)+'fine_fold_' + str(testFold) + '_'+str(cls)+'.pkl')

        ##### Create test set for fine
        data_test = np.asarray(fine_folds[testFold])
        y_test, X_test = data_test[:, 0], data_test[:, 1:data_test.shape[1]]
        y_testCoarse = []
        for i in y_test:
            if i > 0:
                y_testCoarse.append(1.)
            else:
                y_testCoarse.append(0.)

        y_score = []
        for i in range(1, 9):
            clf = joblib.load('fine_models/rnd'+str(rndNum)+'fine_fold_' + str(testFold) + '_'+str(cls)+'.pkl')
            y_preScore = clf.decision_function(X_test)
            y_preScore = y_preScore.reshape(y_preScore.shape[0], 1)
            if (len(y_score) == 0):
                y_score = y_preScore
            else:
                y_score = np.hstack((y_score, y_preScore))

        ##### Predict test set for fine
        y_scoreTmp = []
        for i, score in enumerate(y_score):
            nums = y_score[i]
            nums = list(map(float, nums))
            maxFine = max(nums)
            y_scoreTmp.append(maxFine)
        y_pred_score = np.asarray(y_scoreTmp)


        ##### Print this folds roc_auc for fine
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[1], tpr[1], thresholds = roc_curve(y_testCoarse, y_pred_score)
        roc_auc[1] = auc(fpr[1], tpr[1])
        # Plot of a ROC curve
        plt.figure()
        plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.5f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('results/rnd'+str(rndNum)+'_fine_ROC_' + str(testFold) + '.png')
        plt.clf()
        #plt.close()

        ##### Print this folds pr_curve for fine
        precision = dict()
        recall = dict()
        average_precision = dict()
        precision[1], recall[1], _ = precision_recall_curve(y_testCoarse, y_pred_score)
        average_precision[1] = average_precision_score(y_testCoarse, y_pred_score)
        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[1], precision[1], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall: AUC={0:0.5f}'.format(average_precision[1]))
        plt.legend(loc="lower left")
        # plt.show()
        plt.savefig('results/rnd'+str(rndNum)+'_fine_PR_' + str(testFold) + '.png')
        plt.clf()
        plt.close()
        rnd_results_fine.append([rndNum]+[testFold] + [roc_auc[1]] + [average_precision[1]])




def confEstPopSetsCoarseFine(classes_all,coarse_set,fine_set,rndNum,coarseAdd,fineAdd):
    coarse_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    fine_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    coarse_decFcn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    fine_decFcn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}




    for i in sorted(classes_all):
        partition = np.asarray(classes_all[i])
        data = partition
        y_train, X_train = data[:, 0], data[:, 1:data.shape[1]]
        y_trainCoarse = []
        for i in y_train:
            if i > 0:
                y_trainCoarse.append(1.)
            else:
                y_trainCoarse.append(i)

        for i in range(1, 11):
            coarse_clf = joblib.load('coarse_models/rnd' + str(rndNum) + '_coarse_fold_' + str(i) + '.pkl')
            if(len(coarse_decFcn[i]) == 0):
                coarse_decFcn[i] = list(coarse_clf.decision_function(X_train))
            else:

        rnd_coarse_clf = joblib.load('coarse_models/rnd' + str(rndNum) + '_coarse_fold_'+str(worstCoarse)+'.pkl')
        coarse_decFcn[i] = list(rnd_coarse_clf.decision_function(X_train))
        rnd_fine_clf = joblib.load('fine_models/rnd' + str(rndNum) + '_fine_fold_'+str(worstFine)+'.pkl')
        fine_decFcn[i] = list(rnd_fine_clf.decision_function(X_train))




    for i in range(coarseAdd):
        most_uncert = 100
        most_cls = 0
        most_ind = 0

        for i in sorted(coarse_decFcn):
            for j, inst in enumerate(coarse_decFcn[i]):
                uncert = np.absolute(inst)
                if (uncert < most_uncert):
                    most_cls = i
                    most_ind = j
                    most_uncert = uncert
        coarse_set[most_cls].append(classes_all[most_cls].pop(most_ind))
        del coarse_decFcn[most_cls][most_ind]
        del fine_decFcn[most_cls][most_ind]

    for i in range(fineAdd):
        most_uncert = 100
        most_cls = 0
        most_ind = 0
        for i in sorted(fine_decFcn):
            for j, inst in enumerate(fine_decFcn[i]):
                for eachClassEst in inst:
                    uncert = np.absolute(eachClassEst)
                    if (uncert < most_uncert):
                        most_cls = i
                        most_ind = j
                        most_uncert = uncert
        coarse_set[most_cls].append(classes_all[most_cls][most_ind])
        fine_set[most_cls].append(classes_all[most_cls].pop(most_ind))
        del coarse_decFcn[most_cls][most_ind]
        del fine_decFcn[most_cls][most_ind]

