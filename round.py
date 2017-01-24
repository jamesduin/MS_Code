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
import methods as m

start_time = time.perf_counter()

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}

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
#print(tot)
totVect = tot/np.sum(tot)
#print(totVect)


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


coarse_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
fine_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
##### Create folds for coarse set
m.createFolds(coarse_set, coarse_folds)

##### Create folds for fine set
m.createFolds(fine_set, fine_folds)


rndNum = 1

#####  Iterate through fold list for coarse
m.iterateFoldsCoarse(rndNum, coarse_folds)


##### Iterate through fold list for fine
m.iterateFoldsFine(rndNum, fine_folds)


print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))

start_time = time.perf_counter()



###### run confidence estimate for coarse and fine
m.confEstPopSetsCoarseFine(classes_all,coarse_set,fine_set,rndNum,30,70)


coarse_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
fine_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
##### Create folds for coarse set
m.createFolds(coarse_set, coarse_folds)

##### Create folds for fine set
m.createFolds(fine_set, fine_folds)



rndNum = 2

#####  Iterate through fold list for coarse
m.iterateFoldsCoarse(rndNum, coarse_folds)


##### Iterate through fold list for fine
m.iterateFoldsFine(rndNum, fine_folds)


print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))













