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






#####  Iterate through fold list for coarse
fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = [['fold','roc_auc','pr_auc']]
for testFold in fold_list:
    print("coarse fold"+str(testFold))
    ##### Create train set for coarse
    data = []
    partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partition_list.remove(testFold)
    for x in partition_list:
        partition = np.loadtxt("../data/partition_"+str(x))
        if data == []:
            data = partition
        else:
            data = np.vstack((partition, data))
    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]
    y_trainCoarse = []
    for i in y_train:
        if i > 0:
            y_trainCoarse.append(1.)
        else:
            y_trainCoarse.append(i)

    #### scale dataset
    scaler = preprocessing.StandardScaler().fit(X_trainPreScale);
    X_trainFull = scaler.transform(X_trainPreScale);
    selector = SelectPercentile(f_classif, percentile=75);
    selector.fit(X_trainFull,y_trainCoarse);
    X_train = selector.transform(X_trainFull);


    ##### Create test set for coarse
    data_test = np.loadtxt("../data/partition_"+str(testFold))
    y_test,X_testPreScale = data_test[:,0], data_test[:,1:data_test.shape[1]];
    X_testFull = scaler.transform(X_testPreScale);
    X_test = selector.transform(X_testFull);
    y_testCoarse = []
    for i in y_test:
        if i > 0:
            y_testCoarse.append(1.)
        else:
            y_testCoarse.append(i)


    ##### Train classifier for coarse
    # classifier = svm.SVC(C=10.0, kernel='poly',degree=3, probability=True, cache_size=8192,decision_function_shape='ovr', verbose=False)
    # clf = classifier.fit(X_train,y_trainCoarse)
    # joblib.dump(clf,'coarse_models/coarse_fold_'+str(testFold)+'.pkl')
    clf = joblib.load('coarse_models/coarse_fold_'+str(testFold)+'.pkl')


    ##### Predict test set for coarse
    y_pred = clf.predict(X_test);
    y_predCoarse = []
    for i in y_pred:
        if i > 0:
            y_predCoarse.append(1.)
        else:
            y_predCoarse.append(i)
    y_prob = clf.predict_log_proba(X_test)
    #print(y_prob.shape)
    y_score = []
    for i in range(0,y_prob.shape[0]):
        # print(y_prob[i,:])
        maxVal = max(y_prob[i,:])
        if maxVal == y_prob[i,0]:
            maxVal = 0.
        y_score.append(maxVal-y_prob[i,0])



#     ##### Print this folds roc_auc for coarse
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     fpr[1], tpr[1], thresholds = roc_curve(y_testCoarse,y_score)
#     roc_auc[1] = auc(fpr[1], tpr[1])
#     # Plot of a ROC curve
#     plt.figure()
#     plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.5f)' % roc_auc[1])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     #plt.show()
#     plt.savefig('results/coarse_ROC_' + str(testFold) + '.png')
#
#
#     ##### Print this folds pr_auc for coarse
#     # Compute Precision-Recall and plot curve
#     precision = dict()
#     recall = dict()
#     average_precision = dict()
#     precision[1], recall[1], _ = precision_recall_curve(y_testCoarse, y_score)
#     average_precision[1] = average_precision_score(y_testCoarse, y_score)
#     # Plot Precision-Recall curve
#     plt.clf()
#     plt.plot(recall[1], precision[1], label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('Precision-Recall: AUC={0:0.5f}'.format(average_precision[1]))
#     plt.legend(loc="lower left")
#     #plt.show()
#     plt.savefig('results/coarse_PR_' + str(testFold) + '.png')
#     results.append([str(testFold)]+[roc_auc[1]]+[average_precision[1]]);
# # f = open('results/_coarseResults.txt', 'w')
# # for result in results:
# #     index = 0
# #     for r in result[:-1]:
# #         f.write(str(result[index])+", ")
# #         index += 1
# #     f.write(str(result[-1]))
# #     f.write("\n")
# # f.close()









##### Iterate through fold list for fine
fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = [['fold','roc_auc','pr_auc']]
for testFold in fold_list:
    print("fine fold" + str(testFold))
    ##### Create train set for fine
    data = []
    partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partition_list.remove(testFold)
    for x in partition_list:
        partition = np.loadtxt("../data/partition_"+str(x))
        if data == []:
            data = partition
        else:
            data = np.vstack((partition, data))
        #print(str(x)+"=>" +str(data.shape))
    #print(data.shape)
    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]

    #### Scale dataset
    scaler = preprocessing.StandardScaler().fit(X_trainPreScale);
    X_trainFull = scaler.transform(X_trainPreScale);
    selector = SelectPercentile(f_classif,percentile=75);
    selector.fit(X_trainFull,y_train);
    X_train = selector.transform(X_trainFull);

    ##### Create test set for fine
    data_test = np.loadtxt("../data/partition_"+str(testFold))
    y_test,X_testPreScale = data_test[:,0], data_test[:,1:data_test.shape[1]];
    X_testFull = scaler.transform(X_testPreScale);
    X_test = selector.transform(X_testFull);
    y_testCoarse = []
    for i in y_test:
        if i > 0:
            y_testCoarse.append(1.)
        else:
            y_testCoarse.append(i)

    ##### Train classifier for fine
    # classifier = svm.SVC(C=10.0, kernel='poly',degree=3, probability=True, cache_size=8192,decision_function_shape='ovr', verbose=False)
    # clf = classifier.fit(X_train,y_train)
    # joblib.dump(clf,'fine_models/fine_fold_'+str(testFold)+'.pkl')
    clf = joblib.load('fine_models/fine_fold_'+str(testFold)+'.pkl')


    ##### Predict test set for fine
    y_pred = clf.predict(X_test);
    y_predCoarse = []
    for i in y_pred:
        if i > 0:
            y_predCoarse.append(1.)
        else:
            y_predCoarse.append(i)
    y_prob = clf.predict_log_proba(X_test)
    #print(y_prob.shape)
    y_score = []
    for i in range(0,y_prob.shape[0]):
        # print(y_prob[i,:])
        maxVal = max(y_prob[i,:])
        if maxVal == y_prob[i,0]:
            maxVal = 0.
        y_score.append(maxVal-y_prob[i,0])


#     ##### Print this folds roc_auc for fine
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     fpr[1], tpr[1], thresholds = roc_curve(y_testCoarse,y_score)
#     roc_auc[1] = auc(fpr[1], tpr[1])
#     # Plot of a ROC curve
#     plt.figure()
#     plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.5f)' % roc_auc[1])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     #plt.show()
#     plt.savefig('results/fine_ROC_' + str(testFold) + '.png')
#
#
#     ##### Print this folds pr_curve for fine
#     # Compute Precision-Recall and plot curve
#     precision = dict()
#     recall = dict()
#     average_precision = dict()
#     precision[1], recall[1], _ = precision_recall_curve(y_testCoarse, y_score)
#     average_precision[1] = average_precision_score(y_testCoarse, y_score)
#     # Plot Precision-Recall curve
#     plt.clf()
#     plt.plot(recall[1], precision[1], label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('Precision-Recall: AUC={0:0.5f}'.format(average_precision[1]))
#     plt.legend(loc="lower left")
#     #plt.show()
#     plt.savefig('results/fine_PR_' + str(testFold) + '.png')
#     results.append([str(testFold)]+[roc_auc[1]]+[average_precision[1]]);
# # f = open('results/fineResults.txt', 'w')
# # for result in results:
# #     index = 0
# #     for r in result[:-1]:
# #         f.write(str(result[index])+", ")
# #         index += 1
# #     f.write(str(result[-1]))
# #     f.write("\n")
# # f.close()