import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
from sklearn.multiclass import OneVsRestClassifier
import time
import re
import methods as m
file_name = re.split("[/\.]",__file__)[-2]
level = re.split("_",file_name)[1]

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

def fcnSclWeight(x):
    return 0.685331066*x+6.5884

# using the pre partitioned data
#### store totals
totals = []
for i in sorted(coarse_folds):
    #with open("../data/partition_subset/partition_sub" + str(i)) as f:
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            coarse_folds[i].append(nums)
    np.random.shuffle(coarse_folds[i])
    totals.append(len(coarse_folds[i]))
tot = np.array(totals)
totVect = tot/np.sum(tot)
start_time = time.perf_counter()


#
# #### store totals
# totals = []
# for i in sorted(classes_all):
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
# ##### Create folds for coarse set
# for i in sorted(classes_all):
#     np.random.shuffle(classes_all[i])
#     partList = []
#     for j in sorted(coarse_folds):
#         partList.append((j,len(coarse_folds[j])))
#     minIndex = partList[0][0]
#     minVal = partList[0][1]
#     for j in sorted(partList):
#         if(minVal > j[1] ):
#             minVal = j[1]
#             minIndex = j[0]
#     partitionCounter = minIndex
#     for instance in classes_all[i]:
#         coarse_folds[partitionCounter].append(instance)
#         partitionCounter+=1
#         if partitionCounter > 10:
#             partitionCounter = 1
#
# for i in sorted(coarse_folds):
#     np.random.shuffle(coarse_folds[i])


f = open('results/_'+file_name+'.txt', 'w')
#####  Iterate through fold list for coarse
fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#fold_list = [1]
results_coarse =[]
for testFold in fold_list:
    print(level+" fold"+str(testFold))
    f.write('{} fold {}\n'.format(level,testFold))
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
    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]
    y_trainCoarse = []
    for i in y_train:
        if i > 0:
            y_trainCoarse.append(1.)
        else:
            y_trainCoarse.append(0.)
    train_wt = fcnSclWeight(len(y_train)/np.sum(y_trainCoarse))
    print('train_wt: {}'.format(train_wt))

    #### Scale dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_trainPreScale)




    ##### Train classifier for coarse
    classifier = OneVsRestClassifier(linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                                 fit_intercept=False, intercept_scaling=1, class_weight={1: train_wt},
                                                 solver='liblinear',
                                                 max_iter=1000, n_jobs=-1),n_jobs=-1)
    # classifier = OneVsRestClassifier(svm.SVC(C=1.0, kernel='rbf', probability=False,
    #                     cache_size=8192, verbose=False, class_weight='balanced',
    #                      gamma=0.0025, tol=0.00001, shrinking=True))
    # classifier = OneVsRestClassifier(svm.SVC(C=10.0, kernel='poly',
    #     degree=3, probability=False, cache_size=8192, verbose=False))
    # classifier = svm.SVC(C=10.0, kernel='poly', degree=3, probability=False, cache_size=8192,
    #                      decision_function_shape='ovr', verbose=False)
    # classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
    #                                              fit_intercept=False, intercept_scaling=1, class_weight={1: 30},
    #                                              solver='liblinear',
    #                                              max_iter=1000, n_jobs=-1)
    # classifier = svm.SVC(C=1.0, kernel='rbf', probability=False, cache_size=8192,
    #                      decision_function_shape='ovo', verbose=False, class_weight='balanced',
    #                      gamma=0.0025, tol=0.00001, shrinking=True)

    clf = classifier.fit(X_train,y_trainCoarse)
    joblib.dump(clf,level+'_models/'+file_name+'_'+str(testFold)+'.pkl')
    #clf = joblib.load(level+'_models/'+file_name+'_'+str(testFold)+'.pkl')



    ##### Create test set for coarse
    data_test = np.asarray(coarse_folds[testFold])
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


    ##### Predict test set for coarse
    y_predCoarse = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    print("cumulative")
    f.write("cumulative\n")
    confMatrix = confusion_matrix(y_testCoarse, y_predCoarse)
    print(confMatrix)
    f.write('[{:>4}{:>4}]\n[{:>4}{:>4}]\n'.format(confMatrix[0][0], confMatrix[0][1], confMatrix[1][0], confMatrix[1][1]))
    print(accuracy_score(y_testCoarse, y_predCoarse))
    f.write('{:.3}\n'.format(accuracy_score(y_testCoarse, y_predCoarse)))

    ###### log the errors
    err_file = open('other_results/_'+file_name+'Err.txt', 'w')
    for i,pred in enumerate(y_predCoarse):
        if(y_predCoarse[i] != y_testCoarse[i]):
            m.printDataInstance(np.array([y_test[i]]+list(X_test[i])), err_file)
    err_file.close()

    ###### Print this folds roc_auc for coarse
    fpr, tpr, threshRoc = roc_curve(y_testCoarse,y_score,sample_weight=y_sampleWeight)
    roc_auc = auc(fpr, tpr,reorder=True)
    #roc_auc = roc_auc_score(y_testCoarse, y_score, average='weighted')
    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr, tpr,
             label='ROC curve (area = {0:0.3f})'.format(roc_auc),
             color='red', linestyle=':', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(level+'_results/'+file_name+'_ROC_' + str(testFold) + '.png')



    ###### Print this folds pr_auc for coarse
    precision, recall, threshPr = precision_recall_curve(y_testCoarse, y_score,sample_weight=y_sampleWeight)
    #average_precision = average_precision_score(y_testCoarse, y_score, average='weighted')
    average_precision = auc(recall, precision)
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, color='blue', linestyle=':', lw=2,
             label='Precision-recall curve (area = {0:0.3f})'.format(average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(level+'_results/'+file_name+'_PR_' + str(testFold) + '.png')
    results_coarse.append([str(testFold)]+[roc_auc]+[average_precision])


###### Save coarse results to a file
f.write(level+'\n')
f.write('{0:5}{1:7}{2:7}\n'.format('fold', 'roc', 'pr'))
roc_Sum = 0.0
pr_Sum = 0.0
for result in results_coarse:
    f.write('{0:<5}{1:<7.3f}{2:<7.3f}\n'.format(*result))
    roc_Sum += result[1]
    pr_Sum += result[2]
f.write('{0:},{1:.3f},{2:.3f} \n'.format('avg', (roc_Sum / len(results_coarse)), (pr_Sum / len(results_coarse))))


print('{} sec'.format(round(time.perf_counter() - start_time, 2)))
f.write('{} sec'.format(round(time.perf_counter() - start_time, 2)))
f.close()







