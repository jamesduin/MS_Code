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



def printDict(dictionary):
    instanceCount = 0
    for i in sorted(dictionary):
        instanceCount += len(dictionary[i])
        print('{0:<10}{1:<10}'.format(i, len(dictionary[i])))
    print('{0:<10}{1:<10}'.format('Total', instanceCount))
    print('Shape: {0:<10}\n'.format(len(dictionary[0][0])))



def iterateFoldsCoarse(level,rndNum, coarse_set,test_part,rnd_results_coarse):
    rnd_results_coarse.append(['rnd', 'roc_auc', 'pr_auc','acc'])
    print("rnd"+str(rndNum)+"_"+level)
    ##### Create train set for coarse
    data = []
    for x in sorted(coarse_set):
        partition = np.asarray(coarse_set[x])
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
    data_test = np.asarray(test_part[1])
    y_test, X_test = data_test[:, 0], data_test[:, 1:data_test.shape[1]];
    y_testCoarse = []
    for i in y_test:
        if i > 0:
            y_testCoarse.append(1.)
        else:
            y_testCoarse.append(0.)

    ##### Train classifier for coarse
    classifier = OneVsRestClassifier(linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                                     fit_intercept=False, intercept_scaling=1,
                                                                     class_weight={1: 20},
                                                                     solver='liblinear',
                                                                     max_iter=1000, n_jobs=-1))
    clf = classifier.fit(X_train, y_trainCoarse)
    joblib.dump(clf, level+'_models/rnd'+str(rndNum)+'_'+level+'.pkl')
    # clf = joblib.load(clf, level+'_models/rnd'+str(rndNum)+'_'+level+'.pkl')


    ##### Predict test set for coarse
    y_predCoarse = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    print("cumulative")
    confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
    print(confMatrix)
    print(accuracy_score(y_testCoarse, y_predCoarse))
    rnd_results_coarse.append([confMatrix[0][0]] + [confMatrix[0][1]] + [confMatrix[1][0]] + [confMatrix[1][1]])

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
    plt.savefig(level+'_results/rnd'+str(rndNum)+'_'+level+'_ROC.png')
    plt.clf()

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
    plt.savefig(level+'_results/rnd'+str(rndNum)+'_'+level+'_PR.png')
    plt.clf()
    plt.close()
    rnd_results_coarse.append([rndNum] + [roc_auc[1]] + [average_precision[1]]+[accuracy_score(y_testCoarse, y_predCoarse)])











def iterateFoldsFine(level,rndNum,fine_set,test_part,rnd_results_fine):
    rnd_results_fine.append(['rnd', 'roc_auc', 'pr_auc','acc'])
    print("rnd"+str(rndNum)+"_"+level)
    ##### Create train set for fine
    data = []
    for x in sorted(fine_set):
        partition = np.asarray(fine_set[x])
        if data == []:
            data = partition
        else:
            data = np.vstack((partition, data))
    y_train, X_train = data[:, 0], data[:, 1:data.shape[1]]
    y_trainBin = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    ##### Train classifier for fine
    classifier = OneVsRestClassifier(linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                                     fit_intercept=False, intercept_scaling=1,
                                                                     class_weight={1: 20},
                                                                     solver='liblinear',
                                                                     max_iter=1000, n_jobs=-1))

    clf = classifier.fit(X_train, y_trainBin)
    joblib.dump(clf, level+'_models/rnd'+str(rndNum)+'_'+level+'.pkl')
    # clf = joblib.load(level+'_models/rnd'+str(rndNum)+'_'+level+'.pkl')

    ##### Create test set for fine
    data_test = np.asarray(test_part[1])
    y_test, X_test = data_test[:, 0], data_test[:, 1:data_test.shape[1]]
    y_testCoarse = []
    for i in y_test:
        if i > 0:
            y_testCoarse.append(1.)
        else:
            y_testCoarse.append(0.)

    y_testBin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_score = clf.decision_function(X_test)

    ##### Predict test set for fine
    y_pred_score = np.amax(y_score[:,1:],axis=1)

    y_predCoarse = []
    for inst in y_pred_score:
        if (inst > 0.0):
            y_predCoarse.append(1.)
        else:
            y_predCoarse.append(0.)
    print("cumulative")
    confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
    print(confMatrix)
    print(accuracy_score(y_testCoarse, y_predCoarse))
    rnd_results_fine.append([confMatrix[0][0]] + [confMatrix[0][1]] + [confMatrix[1][0]] + [confMatrix[1][1]])


    ##### Print this folds roc_auc for fine
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_testBin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(level+'_results/rnd'+str(rndNum)+'_'+level+'_ROC.png')
    plt.clf()

    ##### Print this folds pr_curve for fine
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_testBin.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(y_testBin, y_score, average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(level+'_results/rnd'+str(rndNum)+'_'+level+'_PR.png')
    plt.clf()
    plt.close()
    rnd_results_fine.append([rndNum] + [roc_auc["micro"]] + [average_precision["micro"]]+[accuracy_score(y_testCoarse, y_predCoarse)])



def findAddInstance(classes,set,find_inst):
    for cls in sorted(classes):
        for index, inst in enumerate(classes[cls]):
            if (inst == find_inst):
                set[inst[0]].append(classes[cls].pop(index))
                return



def confEstPopSetsCoarseFine(classes_coarse,classes_fine,coarse_set,fine_set,rndNum,coarseAdd,fineAdd):
    data = []
    for x in sorted(classes_coarse):
        partition = np.asarray(classes_coarse[x])
        if (len(partition) > 0):
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
    np.random.shuffle(data)
    for i in range(coarseAdd):
        findAddInstance(classes_coarse,coarse_set,data[i].tolist())

    data = []
    for x in sorted(classes_fine):
        partition = np.asarray(classes_fine[x])
        if (len(partition) > 0):
            if data == []:
                data = partition
            else:
                data = np.vstack((partition, data))
    np.random.shuffle(data)
    for i in range(fineAdd):
        findAddInstance(classes_fine, fine_set, data[i].tolist())


    #
    #
    #
    # coarse_decFcn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    # fine_decFcn = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    #
    # for i in sorted(classes_coarse):
    #     data = np.asarray(classes_coarse[i])
    #     if(len(data)>0):
    #         y_train, X_train = data[:, 0], data[:, 1:]
    #         coarse_clf = joblib.load('coarse_models/rnd' + str(rndNum) + '_coarse.pkl')
    #         scores = coarse_clf.decision_function(X_train)
    #         scores = scores.reshape(scores.shape[0],1)
    #         coarse_decFcn[i] = scores.tolist()
    #
    # for i in sorted(classes_fine):
    #     data = np.asarray(classes_fine[i])
    #     if (len(data) > 0):
    #         y_train, X_train = data[:, 0], data[:, 1:data.shape[1]]
    #         fine_clf = joblib.load('fine_models/rnd' + str(rndNum) + '_fine.pkl')
    #         scores = fine_clf.decision_function(X_train)
    #         scores = np.amax(scores[:, 1:], axis=1)
    #         scores = scores.reshape(scores.shape[0],1)
    #         fine_decFcn[i] = scores.tolist()
    #
    # for i in range(coarseAdd):
    #     most_uncert = 100
    #     most_cls = 0
    #     most_ind = 0
    #
    #     for cls in sorted(coarse_decFcn):
    #         for index, inst in enumerate(coarse_decFcn[cls]):
    #             min_est =np.absolute(inst)
    #             if(min_est<most_uncert):
    #                 most_cls = cls
    #                 most_ind = index
    #                 most_uncert = min_est
    #     #print('coarse {},{},{},{}'.format(i, most_cls, most_ind, len(classes_coarse[most_cls])))
    #     coarse_set[most_cls].append(classes_coarse[most_cls].pop(most_ind))
    #     del coarse_decFcn[most_cls][most_ind]
    #
    # for i in range(fineAdd):
    #     most_uncert = 100
    #     most_cls = 0
    #     most_ind = 0
    #     for cls in sorted(fine_decFcn):
    #         for index, inst in enumerate(fine_decFcn[cls]):
    #             min_est = np.absolute(inst)
    #             if (min_est < most_uncert):
    #                 most_cls = cls
    #                 most_ind = index
    #                 most_uncert = min_est
    #     #print('fine {},{},{},{}'.format(i, most_cls, most_ind,len(classes_fine[most_cls])))
    #     fine_set[most_cls].append(classes_fine[most_cls].pop(most_ind))
    #     del fine_decFcn[most_cls][most_ind]


