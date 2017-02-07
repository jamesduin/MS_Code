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
import methodsActive as m
import pickle
import copy

start_time = time.perf_counter()


classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
test_part = {1: []}
train_part = {2: [],3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
rnd_results_coarse = []
rnd_results_fine = []

# #### store totals
# totals = []
# for i in sorted(classes_all):
#     with open("../data/classes_scaled/class_scaled" + str(i)) as f:
#         for line in f:
#             nums = line.split()
#             nums = list(map(float, nums))
#             classes_all[i].append(nums)
#     np.random.shuffle(classes_all[i])
#     totals.append(len(classes_all[i]))
# tot = np.array(totals)
# print(tot)
# totVect = tot/np.sum(tot)



# using the pre partitioned data
for i in sorted(train_part):
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            train_part[i].append(nums)
    np.random.shuffle(train_part[i])

print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('train','',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(train_part):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(train_part[i])
    for inst in train_part[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(train_part[i])] + classCount
    print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))
print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total',instanceCount,*classCountTot))


for i in sorted(test_part):
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            test_part[i].append(nums)
    np.random.shuffle(test_part[i])


print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('test','',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(test_part):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(test_part[i])
    for inst in test_part[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(test_part[i])] + classCount
    print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))
print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total',instanceCount,*classCountTot))


for i in sorted(train_part):
    for index in range(len(train_part[i])):
        classes_all[train_part[i][index][0]].append(train_part[i][index])


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

classes_fine = copy.deepcopy(classes_all)
classes_coarse = copy.deepcopy(classes_all)


rndNum = 1


print('{0:<10}{1:<10}'.format('classes_coarse', ''))
instanceCount = 0
for i in sorted(classes_coarse):
    instanceCount += len(classes_coarse[i])
    print('{0:<10}{1:<10}'.format(i, len(classes_coarse[i])))
print('{0:<10}{1:<10}'.format('Total', instanceCount))
print('Shape: {0:<10}\n'.format(len(classes_coarse[0][0])))



print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('coarse_folds','',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(coarse_folds):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(coarse_folds[i])
    for inst in coarse_folds[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(coarse_folds[i])] + classCount
    print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))
print('{:<7}{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total',instanceCount,*classCountTot))



#####  Iterate through fold list for coarse
m.iterateFoldsCoarse("coarse",rndNum, coarse_folds,rnd_results_coarse)

##### Iterate through fold list for fine
m.iterateFoldsFine("fine",rndNum, fine_folds,rnd_results_fine)

##### Append round time and fold counts
print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))
rnd_results_coarse.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
rnd_results_fine.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])

instanceCount = 0
fold_cnt = ['coarse_folds']
for i in sorted(coarse_folds):
    instanceCount += len(coarse_folds[i])
    fold_cnt.append(['{0}{1}'.format(i, len(coarse_folds[i]))])
fold_cnt.append(['{0}{1}'.format('Total', instanceCount)])
rnd_results_coarse.append(fold_cnt)

instanceCount = 0
fold_cnt = ['fine_folds']
for i in sorted(fine_folds):
    instanceCount += len(fine_folds[i])
    fold_cnt.append(['{0}{1}'.format(i, len(fine_folds[i]))])
fold_cnt.append(['{0}{1}'.format('Total', instanceCount)])
rnd_results_fine.append(fold_cnt)




#for rndNum in range(2,200):
while(instanceCount > 100):
    start_time = time.perf_counter()
    ###### run confidence estimate for coarse and fine
    m.confEstPopSetsCoarseFine(classes_coarse,classes_fine,coarse_set,fine_set,rndNum,100,100)
    rndNum += 1
    coarse_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    fine_folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    ##### Create folds for coarse set
    m.createFolds(coarse_set, coarse_folds)

    ##### Create folds for fine set
    m.createFolds(fine_set, fine_folds)

    #####  Iterate through fold list for coarse
    m.iterateFoldsCoarse("coarse",rndNum, coarse_folds,rnd_results_coarse)

    ##### Iterate through fold list for fine
    m.iterateFoldsFine("fine",rndNum, fine_folds,rnd_results_fine)

    ##### Append round time and fold counts
    rnd_results_coarse.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])
    rnd_results_fine.append(['Rnd'] + [str(rndNum)] + ['Sec'] + [str(round(time.perf_counter() - start_time, 2))])

    instanceCount = 0
    fold_cnt = ['coarse_folds']
    for i in sorted(coarse_folds):
        instanceCount += len(coarse_folds[i])
        fold_cnt.append(['{0}{1}'.format(i, len(coarse_folds[i]))])
    fold_cnt.append(['{0}{1}'.format('Total', instanceCount)])
    rnd_results_coarse.append(fold_cnt)

    instanceCount = 0
    fold_cnt = ['fine_folds']
    for i in sorted(fine_folds):
        instanceCount += len(fine_folds[i])
        fold_cnt.append(['{0}{1}'.format(i, len(fine_folds[i]))])
    fold_cnt.append(['{0}{1}'.format('Total', instanceCount)])
    rnd_results_fine.append(fold_cnt)

    print('Round {0}: {1} seconds'.format(rndNum,round(time.perf_counter() - start_time, 2)))
    fileName = open('results/rnd_results_fine','wb')
    pickle.dump(rnd_results_fine,fileName)
    fileName.close()
    fileName = open('results/rnd_results_coarse','wb')
    pickle.dump(rnd_results_coarse,fileName)
    fileName.close()





