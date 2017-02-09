import numpy as np
from sklearn.externals import joblib
from sklearn import linear_model
import time
import re
from sklearn.preprocessing import label_binarize
import methodsPsvOOP as m

fName = re.split("[/\.]",__file__)[-2]
lvl = re.split("_",fName)[1]

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
results_fine = []

# load the data
for i in sorted(fine_folds):
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            fine_folds[i].append(nums)
    np.random.shuffle(fine_folds[i])

##### iterate through fold list for fine
# fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# #fold_list = [1]
# for testFold in fold_list:

testFold = 1
start_time = time.perf_counter()
rnd = m.LearnRound(lvl,testFold,fine_folds,fName)

##### create train set
y_train, X_train, min_max_scaler = rnd.createTrainSet()

##### create train_wt (y_train unmodified) for fine
y_trainBin = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6, 7, 8])
y_tot = np.sum(y_trainBin)
wt =m.fcnSclWeight(len(y_train)/y_tot)
train_tune = {1: 1, 2: 0.25, 3: 0.4, 4: 0.5, 5: 0.1, 6: 0.75, 7: 1.5, 8: 0.25}
Fine_wt = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
for i in Fine_wt:
    Fine_wt[i] = train_tune[i]*wt
cls_sums = np.sum(y_trainBin,axis=0)
train_wt = []
for i in Fine_wt:
    train_wt.append(train_tune[i]*cls_sums[i-1])
train_wt = m.fcnSclWeight(len(y_train)/np.sum(train_wt))

##### create test set and sample weight
y_testCoarse, X_test, y_sampleWeight = rnd.createTestSet(min_max_scaler,train_wt)

#### train classifier for fine
classifier = dict()
for cls in range(8):
    classif = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                             fit_intercept=False, intercept_scaling=1, class_weight={1: Fine_wt[cls+1]},
                             solver='liblinear',
                             max_iter=1000, n_jobs=-1)
    clf = classif.fit(X_train, y_trainBin[:,cls])
    joblib.dump(clf, lvl+'_models/'+fName+'_'+str(testFold) + '_'+str(cls+1)+'.pkl')
    classifier[cls] = clf

##### predict test set for fine
y_fine_score = []
for cls in range(8):
    scores = classifier[cls].decision_function(X_test)
    scores = scores.reshape(scores.shape[0],1)
    if y_fine_score == []:
        y_fine_score = scores
    else:
        y_fine_score = np.hstack((y_fine_score,scores))
y_score = np.amax(y_fine_score, axis=1)
y_predCoarse = []
for inst in y_score:
    if (inst > 0.0):
        y_predCoarse.append(1.)
    else:
        y_predCoarse.append(0.)

###### print conf matrix,accuracy and f1_score
rnd.printConfMatrix(y_testCoarse, y_predCoarse)

###### Plot ROC and PR curves
rnd.plotRocPrCurves(y_testCoarse, y_score, y_sampleWeight)

###### Save results to a file
rnd.saveResults(start_time)

