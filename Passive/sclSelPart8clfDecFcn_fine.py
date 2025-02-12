import numpy as np
from sklearn import svm
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
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import time
import re
file_name = re.split("[/\.]",__file__)[-2]
level = re.split("_",file_name)[1]

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}


# using the pre partitioned data
#### store totals
totals = []
for i in sorted(fine_folds):
    with open("../data/partition_subset/partition_sub" + str(i)) as f:
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

    print("y_train shape:"+str(y_train.shape))
    f.write('y_train shape:'+str(y_train.shape)+'\n')

    y_trainSets = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    for cls in sorted(y_trainSets):
        for i in y_train:
            if( i == cls):
                y_trainSets[cls].append(1.)
            else:
                y_trainSets[cls].append(0.)

    y_scalers = []
    y_selectors = []
    for cls in sorted(y_trainSets):
        # print("train fine cls: "+str(cls))
        # f.write('train fine cls: '+str(cls)+'\n')
        y_train = y_trainSets[cls]

        #### scale dataset
        scaler = preprocessing.StandardScaler().fit(X_trainPreScale);
        y_scalers.append(scaler)
        X_trainFull = scaler.transform(X_trainPreScale);
        selector = SelectPercentile(f_classif, percentile=75);
        selector.fit(X_trainFull, y_train);
        y_selectors.append(selector)
        X_train = selector.transform(X_trainFull);


        ##### Train classifier for fine
        classifier = svm.SVC(C=10.0, kernel='poly', degree=3, probability=False, cache_size=8192,
                             decision_function_shape='ovr', verbose=False)
        # classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
        #                                              fit_intercept=False, intercept_scaling=1, class_weight={1: 30},
        #                                              solver='liblinear',
        #                                              max_iter=1000, n_jobs=-1)
        # classifier = svm.SVC(C=1.0, kernel='rbf', probability=False, cache_size=8192,
        #                      decision_function_shape='ovo', verbose=False, class_weight='balanced',
        #                      gamma=0.0025, tol=0.00001, shrinking=True)

        clf = classifier.fit(X_train, y_train)
        joblib.dump(clf, level+'_models/'+file_name+'_'+str(testFold) + '_'+str(cls)+'.pkl')
        # clf = joblib.load(level+'_models/'+file_name+'_'+str(testFold) + '_'+str(cls)+'.pkl')

        print('predict train set'+str(cls))
        f.write('predict train set'+str(cls)+'\n')
        y_pred = clf.predict(X_train)
        confMatrix = confusion_matrix(y_train, y_pred)
        print(confMatrix)
        f.write('[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0], confMatrix[0][1], confMatrix[1][0], confMatrix[1][1]))
        print(accuracy_score(y_train, y_pred))
        f.write('{:.3}\n'.format(accuracy_score(y_train, y_pred)))

        #### Create test set for fine
        data_test = np.asarray(fine_folds[testFold])
        y_test,X_testPreScale = data_test[:,0], data_test[:,1:data_test.shape[1]]
        X_testFull = scaler.transform(X_testPreScale);
        X_test = selector.transform(X_testFull);

        y_testCoarse = []
        for i in y_test:
            if i == cls:
                y_testCoarse.append(1.)
            else:
                y_testCoarse.append(0.)

        print('predict test set'+str(cls))
        f.write('predict test set'+str(cls)+'\n')
        y_pred = clf.predict(X_test)
        confMatrix = confusion_matrix(y_testCoarse, y_pred)
        print(confMatrix)
        f.write('[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0], confMatrix[0][1], confMatrix[1][0], confMatrix[1][1]))
        print(accuracy_score(y_testCoarse, y_pred))
        f.write('{:.3}\n\n'.format(accuracy_score(y_testCoarse, y_pred)))




    y_score = []
    for i in range(1,9):
        ##### Create test set for fine
        data_test = np.asarray(fine_folds[testFold])
        y_test, X_testPreScale = data_test[:, 0], data_test[:, 1:data_test.shape[1]]
        X_testFull = y_scalers[i-1].transform(X_testPreScale);
        X_test = y_selectors[i-1].transform(X_testFull);
        y_testCoarse = []
        for inst in y_test:
            if inst > 0:
                y_testCoarse.append(1.)
            else:
                y_testCoarse.append(0.)


        clf = joblib.load(level+'_models/'+file_name+'_'+str(testFold) + '_'+str(cls)+'.pkl')
        y_preScore = clf.decision_function(X_test)
        y_preScore = y_preScore.reshape(y_preScore.shape[0], 1)
        if(len(y_score)==0):
            y_score = y_preScore
        else:
            y_score = np.hstack((y_score,y_preScore))


    ##### Predict test set for fine
    y_scoreTmp = []
    for i, score in enumerate(y_score):
        nums = y_score[i]
        nums = list(map(float, nums))
        maxFine = max(nums)
        y_scoreTmp.append(maxFine)
    y_pred_score = np.asarray(y_scoreTmp)


    y_pred = []
    for inst in y_pred_score:
        if (inst > 0.0):
            y_pred.append(1.)
        else:
            y_pred.append(0.)
    print("cumulative")
    f.write("cumulative\n")
    print("cumulative")
    f.write("cumulative\n")
    confMatrix = confusion_matrix(y_testCoarse, y_pred)
    print(confMatrix)
    f.write(
        '[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0], confMatrix[0][1], confMatrix[1][0], confMatrix[1][1]))
    print(accuracy_score(y_testCoarse, y_pred))
    f.write('{:.3}\n'.format(accuracy_score(y_testCoarse, y_pred)))


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
    plt.savefig(level+'_results/'+file_name+'_ROC_' + str(testFold) + '.png')

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
    plt.savefig(level+'_results/'+file_name+'_PR_' + str(testFold) + '.png')
    results_fine.append([str(testFold)] + [roc_auc[1]] + [average_precision[1]])

###### Save results to a file
f.write(level+'\n')
# f.write('{0:5}{1:7}{2:7}\n'.format('fold', 'roc', 'pr'))
roc_Sum = 0.0
pr_Sum = 0.0
for result in results_fine:
    #f.write('{0:<5}{1:<7.3f}{2:<7.3f}\n'.format(*result))
    roc_Sum += result[1]
    pr_Sum += result[2]
f.write('{0:<5}{1:<7.3f}{2:<7.3f} \n'.format('avg', (roc_Sum / len(results_fine)), (pr_Sum / len(results_fine))))


print('{} sec'.format(round(time.perf_counter() - start_time, 2)))
f.write('{} sec'.format(round(time.perf_counter() - start_time, 2)))
f.close()




