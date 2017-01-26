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


classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

#### store totals
totals = []
for i in sorted(classes_all):
    with open("../data/classes/class_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes_all[i].append(nums)
    np.random.shuffle(classes_all[i])
    totals.append(len(classes_all[i]))
tot = np.array(totals)
totVect = tot/np.sum(tot)

start_time = time.perf_counter()


##### Create folds for fine set
for i in sorted(classes_all):
    np.random.shuffle(classes_all[i])
    partList = []
    for j in sorted(fine_folds):
        partList.append((j, len(fine_folds[j])))
    minIndex = partList[0][0]
    minVal = partList[0][1]
    for j in sorted(partList):
        if (minVal > j[1]):
            minVal = j[1]
            minIndex = j[0]
    partitionCounter = minIndex
    for instance in classes_all[i]:
        fine_folds[partitionCounter].append(instance)
        partitionCounter+=1
        if partitionCounter > 10:
            partitionCounter = 1




##### Iterate through fold list for fine
fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results_fine = []
for testFold in fold_list:
    print("fine fold" + str(testFold))
    ##### Create train set for fine
    data = []
    partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partition_list.remove(testFold)
    for x in partition_list:
        # partition = np.loadtxt("../data/partition_"+str(x))
        partition = np.asarray(fine_folds[x])
        if data == []:
            data = partition
        else:
            data = np.vstack((partition, data))

    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]

    #### Scale dataset
    scaler = preprocessing.StandardScaler().fit(X_trainPreScale);
    X_trainFull = scaler.transform(X_trainPreScale);
    selector = SelectPercentile(f_classif, percentile=75);
    selector.fit(X_trainFull, y_train);
    X_train = selector.transform(X_trainFull);



    ##### Create test set for fine
    data_test = np.asarray(fine_folds[testFold])
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
    classifier = svm.SVC(C=10.0, kernel='poly', degree=3, probability=True, cache_size=8192, decision_function_shape='ovr',
                         verbose=False)
    clf = classifier.fit(X_train, y_train)
    joblib.dump(clf, 'fine_models/rnd1_fine_fold_' + str(testFold) + '.pkl')
    # clf = joblib.load('fine_models/rnd1_fine_fold_'+str(testFold)+'.pkl')
    y_pred = clf.predict(X_test);

    ##### Predict test set for fine
    y_preScore = clf.decision_function(X_test)
    y_scoreTmp = []
    for i, score in enumerate(y_preScore):
        nums = (y_preScore[i]-y_preScore[i][0])
        nums = list(map(float, nums))
        maxFine = max(nums[1:])
        y_scoreTmp.append(maxFine)
    y_score = np.asarray(y_scoreTmp)


    ##### Print this folds roc_auc for fine
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
    plt.savefig('results/fine_ROC_' + str(testFold) + '.png')

    ##### Print this folds pr_curve for fine
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
    plt.savefig('results/fine_PR_' + str(testFold) + '.png')
    results_fine.append([str(testFold)] + [roc_auc[1]] + [average_precision[1]]);

###### Save results to a file
f = open('results/_fineResults.txt', 'w')
f.write('fine\n')
f.write('{0:5}{1:10}{2:10}\n'.format('fold', 'roc', 'pr'))
roc_Sum = 0.0
pr_Sum = 0.0
for result in results_fine:
    f.write('{0:<5}{1:<10.3f}{2:<10.3f}\n'.format(*result))
    roc_Sum += result[1]
    pr_Sum += result[2]
f.write('{0:<5}{1:<10.3f}{2:<10.3f} \n'.format('avg', (roc_Sum / 10.0), (pr_Sum / 10.0)))
f.close()


print('Round {0}: {1} seconds'.format('fine',round(time.perf_counter() - start_time, 2)))
### Round fine: 698.51 seconds











#
#
# ###### run confidence estimate for coarse
# data = []
# #class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# class_list = [1, 2, 3]
# for x in class_list:
#     partition = np.asarray(classes_all[x])
#     if data == []:
#         data = partition
#     else:
#         data = np.vstack((partition, data))
#         # print(str(x)+"=>" +str(data.shape))
# # print(data.shape)
# y_train, X_trainPreScale = data[:, 0], data[:, 1:data.shape[1]]
# # print(y_train)
# # print(X_trainPreScale)
# y_trainCoarse = []
# for i in y_train:
#     if i > 0:
#         y_trainCoarse.append(1.)
#     else:
#         y_trainCoarse.append(i)
#
#
# #### scale dataset
# scaler = preprocessing.StandardScaler().fit(X_trainPreScale);
# X_trainFull = scaler.transform(X_trainPreScale);
# selector = SelectPercentile(f_classif, percentile=75);
# selector.fit(X_trainFull, y_trainCoarse);
# X_train = selector.transform(X_trainFull);

#
# rnd1_clf = joblib.load('coarse_models/rnd1_coarse_fold_1.pkl')
# X_train_pred = clf.predict(X_train)
# X_train_dec_fcn = clf.decision_function(X_train)
# print('{0:10} {1:10}'.format('Predict','Dec Fcn'))
# for i,pred in enumerate(X_train_pred):
#     print('{0:10} {1:10}'.format(X_train_pred[i], X_train_dec_fcn[i]))
#







