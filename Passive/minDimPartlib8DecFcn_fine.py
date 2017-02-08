import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import time
import re
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import methods as m

file_name = re.split("[/\.]",__file__)[-2]
level = re.split("_",file_name)[1]

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}


# using the pre partitioned data
#### store totals
totals = []
for i in sorted(fine_folds):
    #with open("../data/partition_subset/partition_sub" + str(i)) as f:
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            fine_folds[i].append(nums)
    np.random.shuffle(fine_folds[i])
    totals.append(len(fine_folds[i]))
tot = np.array(totals)
totVect = tot/np.sum(tot)
start_time = time.perf_counter()


# #### store totals
# totals = []
# for i in sorted(classes_all):
#     #with open("../data/classes_subsetscld/class_subsetscld" + str(i)) as f:
#     with open("../data/classes_subset/class_" + str(i)) as f:
#         for line in f:
#             nums = line.split()
#             nums = list(map(float, nums))
#             classes_all[i].append(nums)
#     np.random.shuffle(classes_all[i])
#     totals.append(len(classes_all[i]))
# tot = np.array(totals)
# totVect = tot/np.sum(tot)
#
# start_time = time.perf_counter()
#
#
# ##### Create folds for fine set
# for i in sorted(classes_all):
#     np.random.shuffle(classes_all[i])
#     partList = []
#     for j in sorted(fine_folds):
#         partList.append((j, len(fine_folds[j])))
#     minIndex = partList[0][0]
#     minVal = partList[0][1]
#     for j in sorted(partList):
#         if (minVal > j[1]):
#             minVal = j[1]
#             minIndex = j[0]
#     partitionCounter = minIndex
#     for instance in classes_all[i]:
#         fine_folds[partitionCounter].append(instance)
#         partitionCounter+=1
#         if partitionCounter > 10:
#             partitionCounter = 1
# for i in sorted(fine_folds):
#     np.random.shuffle(fine_folds[i])
#
# print('{0:<10}{1:<10}'.format('Classes',''))
# instanceCount = 0
# for i in sorted(classes_all):
#     instanceCount += len(classes_all[i])
#     print('{0:<10}{1:<10}'.format(i,len(classes_all[i])))
# print('{0:<10}{1:<10}\n'.format('Total',instanceCount))


print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Fine','Folds',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(fine_folds):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(fine_folds[i])
    for inst in fine_folds[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(fine_folds[i])] + classCount
    print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))
print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total',instanceCount,*classCountTot))




f = open('results/_'+file_name+'.txt', 'w')
##### Iterate through fold list for fine
#fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fold_list = [1]
results_fine = []
for testFold in fold_list:
    print(level+" fold" + str(testFold))
    f.write('{} fold {}\n'.format(level,testFold))
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
    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]

    y_trainBin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    y_tot = np.sum(y_trainBin,axis=0)
    train_wt =len(y_train)/y_tot
    print(train_wt)

    #### Scale dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_trainPreScale)

    classifier = dict()
    for cls in range(8):
        classif = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                 fit_intercept=False, intercept_scaling=1, class_weight={1: train_wt[cls]},
                                 solver='liblinear',
                                 max_iter=1000, n_jobs=-1)
        clf = classif.fit(X_train, y_trainBin[:,cls])
        joblib.dump(clf, level+'_models/'+file_name+'_'+str(testFold) + '_'+str(cls)+'.pkl')
        # clf = joblib.load(level+'_models/'+file_name+'_'+str(testFold) + '_'+str(cls)+'.pkl')
        classifier[cls] = clf


    ##### Create test set for coarse
    data_test = np.asarray(fine_folds[testFold])
    y_test, X_testPreScale = data_test[:, 0], data_test[:, 1:data_test.shape[1]]
    X_test = min_max_scaler.transform(X_testPreScale)

    y_testCoarse = []
    for inst in y_test:
        if inst > 0:
            y_testCoarse.append(1.0)
        else:
            y_testCoarse.append(0.0)

    test_wt = len(y_test) / np.sum(y_testCoarse)
    print('test_wt: {}'.format(test_wt))
    y_sampleWeight = []
    for inst in y_testCoarse:
        if inst > 0:
            y_sampleWeight.append(test_wt)
        else:
            y_sampleWeight.append(1.0)

    y_score = []
    for cls in range(8):
        scores = classifier[cls].decision_function(X_test)
        scores = scores.reshape(scores.shape[0],1)
        if y_score == []:
            y_score = scores
        else:
            y_score = np.hstack((y_score,scores))

    ##### Predict test set for fine
    y_pred_score = np.amax(y_score, axis=1)

    y_predCoarse = []
    for inst in y_pred_score:
        if (inst > 0.0):
            y_predCoarse.append(1.)
        else:
            y_predCoarse.append(0.)
    print("cumulative")
    f.write("cumulative\n")
    confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
    print(confMatrix)
    f.write('[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0],confMatrix[0][1],confMatrix[1][0],confMatrix[1][1]))
    print(accuracy_score(y_testCoarse, y_predCoarse))
    f.write('{:.3}\n'.format(accuracy_score(y_testCoarse, y_predCoarse)))


    ###### log the errors
    err_file = open('other_results/_'+file_name+'Err.txt', 'w')
    for i,pred in enumerate(y_predCoarse):
        if(y_predCoarse[i] != y_testCoarse[i]):
            m.printDataInstance(np.array([y_test[i]]+list(X_test[i])), err_file)
    err_file.close()


    ##### Print this folds roc_auc for fine
    #fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score)
    fpr, tpr, threshRoc = roc_curve(y_testCoarse, y_pred_score,sample_weight=y_sampleWeight)
    roc_auc = auc(fpr, tpr,reorder=True)
    #roc_auc = roc_auc_score(y_testCoarse,y_pred_score,average='weighted')

    # Plot of a ROC curve
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
    plt.savefig(level+'_results/'+file_name+'_ROC_' + str(testFold) + '.png')

    ##### Print this folds pr_curve for fine
    precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_pred_score,sample_weight=y_sampleWeight)
    #average_precision = average_precision_score(y_testCoarse, y_pred_score, average='weighted')
    average_precision = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, color='blue', lw=2, linestyle=':',
             label='Precision-recall curve (area = {0:0.3f})'.format(average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig(level+'_results/'+file_name+'_PR_' + str(testFold) + '.png')
    results_fine.append([str(testFold)] + [roc_auc] + [average_precision])

###### Save results to a file
#f.write(level+'\n')
# f.write('{0:5}{1:7}{2:7}\n'.format('fold', 'roc', 'pr'))
roc_Sum = 0.0
pr_Sum = 0.0
for result in results_fine:
    #f.write('{0:<5}{1:<7.3f}{2:<7.3f}\n'.format(*result))
    roc_Sum += result[1]
    pr_Sum += result[2]
f.write('{0:},{1:.3f},{2:.3f} \n'.format('avg', (roc_Sum / len(results_fine)), (pr_Sum / len(results_fine))))


print('{} sec'.format(round(time.perf_counter() - start_time, 2)))
f.write('{} sec'.format(round(time.perf_counter() - start_time, 2)))
f.close()




