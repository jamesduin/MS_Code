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
import time


classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

#### store totals
totals = []
for i in sorted(classes_all):
    with open("../data/classes_scaled/class_scaled" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes_all[i].append(nums)
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
for i in sorted(fine_folds):
    np.random.shuffle(fine_folds[i])

print('{0:<10}{1:<10}'.format('Classes',''))
instanceCount = 0
for i in sorted(classes_all):
    instanceCount += len(classes_all[i])
    print('{0:<10}{1:<10}'.format(i,len(classes_all[i])))
print('{0:<10}{1:<10}\n'.format('Total',instanceCount))


print('{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Fine','Folds',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(fine_folds):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(fine_folds[i])
    for inst in fine_folds[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(fine_folds[i])] + classCount
    print('{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))

print('{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total',instanceCount,*classCountTot))





##### Iterate through fold list for fine
fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#fold_list = [1]
results_fine = []
for testFold in fold_list:
    print("fine fold" + str(testFold))
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
    y_train,X_train = data[:,0], data[:,1:data.shape[1]]

    print("y_train shape:"+str(y_train.shape))

    y_trainSets = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    for cls in sorted(y_trainSets):
        for i in y_train:
            if( i == cls):
                y_trainSets[cls].append(1.)
            else:
                y_trainSets[cls].append(0.)


    for cls in sorted(y_trainSets):
        print("train fine cls: "+str(cls))
        y_train = y_trainSets[cls]

        ##### Train classifier for fine
        classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                     fit_intercept=False, intercept_scaling=1, class_weight={1: 30},
                                                     solver='liblinear',
                                                     max_iter=1000, n_jobs=-1)

        clf = classifier.fit(X_train, y_train)
        joblib.dump(clf, 'fine_models/fine_fold_' + str(testFold) + '_'+str(cls)+'.pkl')
        # clf = joblib.load('fine_models/fine_fold_' + str(testFold) + '_'+str(cls)+'.pkl')
        score = clf.score(X_train,y_train)
        print(score)
        y_pred = clf.predict(X_train)
        print(confusion_matrix(y_train, y_pred))

        #### Create test set for fine
        data_test = np.asarray(fine_folds[testFold])
        y_test,X_test = data_test[:,0], data_test[:,1:data_test.shape[1]]

        y_testCoarse = []
        for i in y_test:
            if i == cls:
                y_testCoarse.append(1.)
            else:
                y_testCoarse.append(0.)

        score = clf.score(X_test, y_testCoarse)
        print(score)
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_testCoarse, y_pred))




    ##### Create test set for fine
    data_test = np.asarray(fine_folds[testFold])
    y_test,X_test = data_test[:,0], data_test[:,1:data_test.shape[1]]
    y_testCoarse = []
    for i in y_test:
        if i > 0:
            y_testCoarse.append(1.)
        else:
            y_testCoarse.append(0.)

    y_score = []
    for i in range(1,9):
        clf = joblib.load('fine_models/fine_fold_' + str(testFold) + '_' + str(i) + '.pkl')
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
    print(confusion_matrix(y_testCoarse, y_pred))
    print(accuracy_score(y_testCoarse, y_pred))


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
    plt.savefig('results/fine_ROC_' + str(testFold) + '.png')

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
    plt.savefig('results/fine_PR_' + str(testFold) + '.png')
    results_fine.append([str(testFold)] + [roc_auc[1]] + [average_precision[1]])

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






