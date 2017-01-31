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
rnd_results_coarse = []
rnd_results_fine = []

#### store totals
totals = []
for i in sorted(classes_all):
    with open("data/classes_scaled/class_scaled" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes_all[i].append(nums)
    np.random.shuffle(classes_all[i])
    totals.append(len(classes_all[i]))
tot = np.array(totals)
print(tot)
totVect = tot/np.sum(tot)




print('{0:<10}{1:<10}'.format('Classes', ''))
instanceCount = 0
for i in sorted(classes_all):
    instanceCount += len(classes_all[i])
    print('{0:<10}{1:<10}'.format(i, len(classes_all[i])))
print('{0:<10}{1:<10}'.format('Total', instanceCount))
print('Shape: {0:<10}\n'.format(len(classes_all[0][0])))






#### randomly add to starter sets
start = [30,10,11,11,11,10,10,10,10]
print(start)
print("Sum start => "+str(np.sum(start)))
for i in sorted(classes_all):
    for j in range(int(start[i])):
        inst = classes_all[i].pop()
        coarse_set[i].append(inst)
        fine_set[i].append(inst)

coarse_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
fine_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

##### Create folds for coarse set
m.createFolds(coarse_set, coarse_folds)

##### Create folds for fine set
m.createFolds(fine_set, fine_folds)




rndNum = 1

#####  Iterate through fold list for coarse
m.iterateFoldsCoarse(rndNum, coarse_folds,rnd_results_coarse)

##### Iterate through fold list for fine
m.iterateFoldsFine(rndNum, fine_folds,rnd_results_fine)

print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))
rnd_results_coarse.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
rnd_results_fine.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])




for rndNum in range(2,100):
    start_time = time.perf_counter()

    ###### run confidence estimate for coarse and fine
    m.confEstPopSetsCoarseFine(classes_all,coarse_set,fine_set,(rndNum-1),30,70)

    coarse_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    fine_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    ##### Create folds for coarse set
    m.createFolds(coarse_set, coarse_folds)

    ##### Create folds for fine set
    m.createFolds(fine_set, fine_folds)

    #####  Iterate through fold list for coarse
    m.iterateFoldsCoarse(rndNum, coarse_folds,rnd_results_coarse)

    ##### Iterate through fold list for fine
    m.iterateFoldsFine(rndNum, fine_folds,rnd_results_fine)


    rnd_results_coarse.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
    rnd_results_fine.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
    ###### Save coarse results to a file
    f = open('results/_coarseResults.txt', 'w')
    for result in rnd_results_coarse:
        if (type(result[2]) == str):
            f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
        else:
            f.write('{0:<5}{1:<5}{2:<10.3f}{3:<10.3f} \n'.format(*result))
    f.write(str(rnd_results_coarse))
    f.close()

    ###### Save results to a file
    f = open('results/_fineResults.txt', 'w')
    for result in rnd_results_fine:
        if (type(result[2]) == str):
            f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
        else:
            f.write('{0:<5}{1:<5}{2:<10.3f}{3:<10.3f} \n'.format(*result))
    f.write(str(rnd_results_fine))
    f.close()

    print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))






