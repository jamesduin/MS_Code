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
import methodsActPass as m
import pickle
import copy

start_time = time.perf_counter()

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
folds = {1: [], 2: [],3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
sets = dict
sets['coarse'] = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
sets['fine'] = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
results = dict()

# using the pre partitioned data
for i in sorted(folds):
    with open("../data/partition_scaled/partition_scaled" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            folds[i].append(nums)
    np.random.shuffle(folds[i])



testFold = 1
partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
partition_list.remove(testFold)
for i in sorted(partition_list):
    for index in range(len(folds[i])):
        classes_all[folds[i][index][0]].append(folds[i][index])



#### randomly add to starter sets
start = [30,10,11,11,11,10,10,10,10]
print(start)
print("Sum start => "+str(np.sum(start)))
for i in sorted(classes_all):
    for j in range(int(start[i])):
        inst = classes_all[i].pop()
        for lv in sets:
            sets[lv][i].append(inst)

classes_fine = copy.deepcopy(classes_all)
classes_coarse = copy.deepcopy(classes_all)


rndNum = 1

#####  Iterate through fold list for coarse and fine
m.iterateFolds(rndNum, sets,folds[testFold],results)



##### Append round time and fold counts
print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))
rnd_results_coarse.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
rnd_results_fine.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
instanceCount = 0
fold_cnt = ['coarse_set']
for i in sorted(coarse_set):
    instanceCount += len(coarse_set[i])
    fold_cnt.append(['({0},{1})'.format(i, len(coarse_set[i]))])
fold_cnt.append(['({0},{1})'.format('Total', instanceCount)])
rnd_results_coarse.append(fold_cnt)

instanceCount = 0
fold_cnt = ['fine_set']
for i in sorted(fine_set):
    instanceCount += len(fine_set[i])
    fold_cnt.append(['({0},{1})'.format(i, len(coarse_set[i]))])
fold_cnt.append(['({0},{1})'.format('Total', instanceCount)])
rnd_results_fine.append(fold_cnt)






while((18088-instanceCount) > 100):
    start_time = time.perf_counter()
    ###### run confidence estimate for coarse and fine
    m.confEstPopSetsCoarseFineActive(classes_coarse,classes_fine,coarse_set,fine_set,rndNum,100,100)
    #m.confEstPopSetsCoarseFinePassive(classes_coarse, classes_fine, coarse_set, fine_set, rndNum, 100, 100)
    rndNum += 1

    #####  Iterate through fold list for coarse
    m.iterateFoldsCoarse("coarse",rndNum, coarse_set,test_part,rnd_results_coarse)

    ##### Iterate through fold list for fine
    m.iterateFoldsFine("fine",rndNum, fine_set,test_part,rnd_results_fine)

    ##### Append round time and fold counts
    rnd_results_coarse.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
    rnd_results_fine.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
    instanceCount = 0
    fold_cnt = ['coarse_set']
    for i in sorted(coarse_set):
        instanceCount += len(coarse_set[i])
        fold_cnt.append(['({0},{1})'.format(i, len(coarse_set[i]))])
    fold_cnt.append(['({0},{1})'.format('Total', instanceCount)])
    rnd_results_coarse.append(fold_cnt)

    instanceCount = 0
    fold_cnt = ['fine_set']
    for i in sorted(fine_set):
        instanceCount += len(fine_set[i])
        fold_cnt.append(['({0},{1})'.format(i, len(coarse_set[i]))])
    fold_cnt.append(['({0},{1})'.format('Total', instanceCount)])
    rnd_results_fine.append(fold_cnt)

    print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))
    fileName = open('results/rnd_results_fine','wb')
    pickle.dump(rnd_results_fine,fileName)
    fileName.close()
    fileName = open('results/rnd_results_coarse','wb')
    pickle.dump(rnd_results_coarse,fileName)
    fileName.close()














