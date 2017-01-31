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



classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

#### store totals
totals = []
for i in sorted(classes_all):
    with open("../data/classes_part/class_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes_all[i].append(nums)
    np.random.shuffle(classes_all[i])
    totals.append(len(classes_all[i]))
tot = np.array(totals)
#print(tot)
totVect = tot/np.sum(tot)
#print(totVect)



# print("//////////////// Classes ////////////////")
# instanceCount = 0
# for i in sorted(classes_all):
#     instanceCount += len(classes_all[i])
#     print(str(i)+","+str(len(classes_all[i])) )
# print( "Total => " + str(instanceCount) )



#### randomly add 36 to starter coarse set
coarseStart = np.ceil(totVect*30)
print(coarseStart)
print("Sum coarseStart => "+str(np.sum(coarseStart)))
for i in sorted(classes_all):
    for j in range(int(coarseStart[i])):
        coarse_set[i].append(classes_all[i].pop(j))




#### randomly add 74 to starter fine set
fineStart = np.ceil(totVect*70)
print(fineStart)
print("Sum fineStart => "+str(np.sum(fineStart)))
for i in sorted(classes_all):
    for j in range(int(fineStart[i])):
        fine_set[i].append(classes_all[i].pop(j))



#
# print("//////////////// Classes ////////////////")
# instanceCount = 0
# for i in sorted(classes_all):
#     instanceCount += len(classes_all[i])
#     print(str(i)+","+str(len(classes_all[i])) )
# print( "Total => " + str(instanceCount) )
#
#
#
# print("//////////////// Coarse Set ////////////////")
# for i in sorted(coarse_set):
#     print(str(i) + "," + str(len(coarse_set[i])))
#     for inst in coarse_set[i]:
#         print(inst[0:6])
#
# print("//////////////// Fine Set ////////////////")
# for i in sorted(fine_set):
#     print(str(i) + "," + str(len(fine_set[i])))
#     for inst in fine_set[i]:
#         print(inst[0:6])





##### Create folds for coarse set
for i in sorted(coarse_set):
    np.random.shuffle(coarse_set[i])
    partList = []
    for j in sorted(coarse_folds):
        partList.append((j,len(coarse_folds[j])))
    minIndex = partList[0][0]
    minVal = partList[0][1]
    for j in sorted(partList):
        if(minVal > j[1] ):
            minVal = j[1]
            minIndex = j[0]
    partitionCounter = minIndex
    for instance in coarse_set[i]:
        coarse_folds[partitionCounter].append(instance)
        partitionCounter+=1
        if partitionCounter > 10:
            partitionCounter = 1

##### Create folds for fine set
for i in sorted(fine_set):
    np.random.shuffle(fine_set[i])
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
    for instance in fine_set[i]:
        fine_folds[partitionCounter].append(instance)
        partitionCounter+=1
        if partitionCounter > 10:
            partitionCounter = 1

# print("//////////////// Coarse Folds ////////////////")
# for i in sorted(coarse_folds):
#     print(str(i) + "," + str(len(coarse_folds[i])))
#     for inst in coarse_folds[i]:
#         print(inst[0:6])
#
# print("//////////////// Fine Folds ////////////////")
# for i in sorted(fine_folds):
#     print(str(i) + "," + str(len(fine_folds[i])))
#     for inst in fine_folds[i]:
#         print(inst[0:6])













#####  Iterate through fold list for coarse
#fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fold_list = [1]
results_coarse = [['fold','roc_auc','pr_auc']]
for testFold in fold_list:
    print("rnd1_coarse fold"+str(testFold))
    ##### Create train set for coarse
    data = []
    partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partition_list.remove(testFold)
    for x in partition_list:
        #partition = np.loadtxt("../data/partition_"+str(x))
        #partition = np.loadtxt(coarse_folds[x])
        partition = np.asarray(coarse_folds[x])
        if data == []:
            data = partition
        else:
            data = np.vstack((partition, data))
        #print(str(x)+"=>" +str(data.shape))
    #print(data.shape)
    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]
    #print(y_train)
    #print(X_trainPreScale)
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
    selector.fit(X_trainFull, y_trainCoarse);
    X_train = selector.transform(X_trainFull);

    ##### Create test set for coarse
    #data_test = np.loadtxt("../data/partition_" + str(testFold))
    data_test = np.asarray(coarse_folds[testFold])
    y_test, X_testPreScale = data_test[:, 0], data_test[:, 1:data_test.shape[1]];
    X_testFull = scaler.transform(X_testPreScale);
    X_test = selector.transform(X_testFull);
    y_testCoarse = []
    for i in y_test:
        if i > 0:
            y_testCoarse.append(1.)
        else:
            y_testCoarse.append(i)


    ##### Train classifier for coarse
    classifier = svm.SVC(C=10.0, kernel='poly',degree=3, probability=False, cache_size=8192,decision_function_shape='ovr', verbose=False)
    clf = classifier.fit(X_train,y_trainCoarse)
    joblib.dump(clf,'coarse_models/rnd1_coarse_fold_'+str(testFold)+'.pkl')
    #clf = joblib.load('coarse_models/rnd1_coarse_fold_'+str(testFold)+'.pkl')
    y_pred = clf.predict(X_test);


    ##### Predict test set for coarse
    y_predCoarse = []
    for i in y_pred:
        if i > 0:
            y_predCoarse.append(1.)
        else:
            y_predCoarse.append(i)
    # Compute confusion matrix
    # cm = confusion_matrix(y_testCoarse, y_predCoarse)
    # np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    # print('Precision / Recall')
    # print(precision_score(y_testCoarse, y_predCoarse, average='binary'))
    # print(recall_score(y_testCoarse, y_predCoarse, average='binary'))
    y_prob = clf.predict_log_proba(X_test)
    #print(y_prob.shape)
    y_score = []
    for i in range(0,y_prob.shape[0]):
        # print(y_prob[i,:])
        maxVal = max(y_prob[i,:])
        if maxVal == y_prob[i,0]:
            maxVal = 0.
        y_score.append(maxVal-y_prob[i,0])

    ###### Print this folds roc_auc for coarse
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[1], tpr[1], thresholds = roc_curve(y_testCoarse,y_score)
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
    #plt.show()
    plt.savefig('results/rnd1_coarse_ROC_' + str(testFold) + '.png')



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
    #plt.show()
    plt.savefig('results/rnd1_coarse_PR_' + str(testFold) + '.png')
    results_coarse.append([str(testFold)]+[roc_auc[1]]+[average_precision[1]]);


###### Save coarse results to a file
f = open('results/rnd1_coarseResults.txt', 'w')
for result in results_coarse:
    index = 0
    for r in result[:-1]:
        f.write(str(result[index])+", ")
        index += 1
    f.write(str(result[-1]))
    f.write("\n")
f.close()





















##### Iterate through fold list for fine
#fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fold_list = [1]
results_fine = [['fold','roc_auc','pr_auc']]
for testFold in fold_list:
    print("rnd1_fine fold" + str(testFold))
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
        #print(str(x)+"=>" +str(data.shape))
    #print(data.shape)
    y_train,X_trainPreScale = data[:,0], data[:,1:data.shape[1]]

    #### Scale dataset
    scaler = preprocessing.StandardScaler().fit(X_trainPreScale);
    X_trainFull = scaler.transform(X_trainPreScale);
    selector = SelectPercentile(f_classif, percentile=75);
    selector.fit(X_trainFull, y_train);
    X_train = selector.transform(X_trainFull);



    ##### Create test set for fine
    #data_test = np.loadtxt("../data/partition_"+str(testFold))
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
    classifier = svm.SVC(C=10.0, kernel='poly', degree=3, probability=False, cache_size=8192, decision_function_shape='ovr',
                         verbose=False)
    clf = classifier.fit(X_train, y_train)
    joblib.dump(clf, 'fine_models/rnd1_fine_fold_' + str(testFold) + '.pkl')
    # clf = joblib.load('fine_models/rnd1_fine_fold_'+str(testFold)+'.pkl')
    y_pred = clf.predict(X_test);

    ##### Predict test set for fine
    y_predCoarse = []
    for i in y_pred:
        if i > 0:
            y_predCoarse.append(1.)
        else:
            y_predCoarse.append(i)
    # Compute confusion matrix
    # cm = confusion_matrix(y_testCoarse, y_predCoarse)
    # np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    # print('Precision / Recall')
    # print(precision_score(y_testCoarse, y_predCoarse, average='binary'))
    # print(recall_score(y_testCoarse, y_predCoarse, average='binary'))
    y_prob = clf.predict_log_proba(X_test)
    # print(y_prob.shape)
    y_score = []
    for i in range(0, y_prob.shape[0]):
        # print(y_prob[i,:])
        maxVal = max(y_prob[i, :])
        if maxVal == y_prob[i, 0]:
            maxVal = 0.
        y_score.append(maxVal - y_prob[i, 0])


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
    plt.savefig('results/rnd1_fine_ROC_' + str(testFold) + '.png')

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
    plt.savefig('results/rnd1_fine_PR_' + str(testFold) + '.png')
    results_fine.append([str(testFold)] + [roc_auc[1]] + [average_precision[1]]);

###### Save results to a file
f = open('results/rnd1_fineResults.txt', 'w')
for result in results_fine:
    index = 0
    for r in result[:-1]:
        f.write(str(result[index])+", ")
        index += 1
    f.write(str(result[-1]))
    f.write("\n")
f.close()

















###### run confidence estimate for coarse
data = []
#class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
class_list = [1, 2, 3]
for x in class_list:
    partition = np.asarray(classes_all[x])
    if data == []:
        data = partition
    else:
        data = np.vstack((partition, data))
        # print(str(x)+"=>" +str(data.shape))
# print(data.shape)
y_train, X_trainPreScale = data[:, 0], data[:, 1:data.shape[1]]
# print(y_train)
# print(X_trainPreScale)
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
selector.fit(X_trainFull, y_trainCoarse);
X_train = selector.transform(X_trainFull);


rnd1_clf = joblib.load('coarse_models/rnd1_coarse_fold_1.pkl')
X_train_pred = clf.predict(X_train)
X_train_dec_fcn = clf.decision_function(X_train)
print('{0:10} {1:10}'.format('Predict','Dec Fcn'))
for i,pred in enumerate(X_train_pred):
    print('{0:10} {1:10}'.format(X_train_pred[i], X_train_dec_fcn[i]))








