import numpy as np
import time
import methodsActPass as m
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


# using the pre partitioned data
for i in sorted(train_part):
    with open("../data/partition_scaled/partition_scaled" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            train_part[i].append(nums)
    np.random.shuffle(train_part[i])
m.printClsVsFolds(train_part, 'train')
for i in sorted(test_part):
    with open("../data/partition_scaled/partition_scaled" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            test_part[i].append(nums)
    np.random.shuffle(test_part[i])
m.printClsVsFolds(test_part, 'test')
for i in sorted(train_part):
    for index in range(len(train_part[i])):
        classes_all[train_part[i][index][0]].append(train_part[i][index])

m.printClassTotals(classes_all)

#### randomly add to starter sets
start = [30,10,11,11,11,10,10,10,10]
print(start)
print("Sum start => "+str(np.sum(start)))
for i in sorted(classes_all):
    for j in range(int(start[i])):
        inst = classes_all[i].pop()
        coarse_set[i].append(inst)
        fine_set[i].append(inst)

classes_fine = copy.deepcopy(classes_all)
classes_coarse = copy.deepcopy(classes_all)

rndNum = 1

#####  Iterate through fold list for coarse
rnd_results_coarse.append(['rnd', 'roc_auc', 'pr_auc', 'acc'])
print("rnd" + str(rndNum) + "_" + "coarse")
rndCrs = m.CoarseRound()
rndCrs.rndNum = rndNum
y_train,X_train = rndCrs.createTrainSet(coarse_set)
y_trainCoarse,train_wt = rndCrs.createTrainWtYtrain(y_train)
y_testCoarse, y_sampleWeight, X_test = rndCrs.createTestSet(test_part,train_wt)
rndCrs.trainClassifier(X_train,y_trainCoarse)
y_predCoarse,y_pred_score = rndCrs.predictTestSet(X_test)
rndCrs.printConfMatrix(y_testCoarse,y_predCoarse,rnd_results_coarse)
rndCrs.plotRocPrCurves(y_testCoarse,y_pred_score,y_sampleWeight,rnd_results_coarse)

##### Iterate through fold list for fine
rnd_results_fine.append(['rnd', 'roc_auc', 'pr_auc','acc'])
print("rnd" + str(rndNum) + "_" + "fine")
rndFin = m.FineRound()
rndFin.rndNum = rndNum
y_train,X_train = rndFin.createTrainSet(fine_set)
y_trainBin,train_wt = rndFin.createTrainWtYtrain(y_train)
y_testCoarse, y_sampleWeight, X_test = rndFin.createTestSet(test_part,train_wt)
rndFin.trainClassifier(X_train,y_trainBin)
y_predCoarse,y_pred_score = rndFin.predictTestSet(X_test)
rndFin.printConfMatrix(y_testCoarse,y_predCoarse,rnd_results_fine)
rndFin.plotRocPrCurves(y_testCoarse,y_pred_score,y_sampleWeight,rnd_results_fine)

##### Append round time and fold counts
m.appendRndTimesFoldCnts(rndNum,'coarse',rnd_results_coarse,coarse_set,start_time)
m.appendRndTimesFoldCnts(rndNum,'fine',rnd_results_fine,fine_set,start_time)

#while((18088-instanceCount) > 100):
while(rndNum < 3):
    start_time = time.perf_counter()
    ###### run confidence estimate for coarse and fine
    m.confEstAdd(classes_coarse,coarse_set,rndCrs,100)
    m.confEstAdd(classes_fine, fine_set, rndFin, 100)

    rndNum += 1

    #####  Iterate through fold list for coarse
    rnd_results_coarse.append(['rnd', 'roc_auc', 'pr_auc', 'acc'])
    print("rnd" + str(rndNum) + "_" + "coarse")
    rndCrs = m.CoarseRound()
    rndCrs.rndNum = rndNum
    y_train, X_train = rndCrs.createTrainSet(coarse_set)
    y_trainCoarse, train_wt = rndCrs.createTrainWtYtrain(y_train)
    y_testCoarse, y_sampleWeight, X_test = rndCrs.createTestSet(test_part, train_wt)
    rndCrs.trainClassifier(X_train, y_trainCoarse)
    y_predCoarse, y_pred_score = rndCrs.predictTestSet(X_test)
    rndCrs.printConfMatrix(y_testCoarse, y_predCoarse, rnd_results_coarse)
    rndCrs.plotRocPrCurves(y_testCoarse, y_pred_score, y_sampleWeight, rnd_results_coarse)

    ##### Iterate through fold list for fine
    rnd_results_fine.append(['rnd', 'roc_auc', 'pr_auc', 'acc'])
    print("rnd" + str(rndNum) + "_" + "fine")
    rndFin = m.FineRound()
    rndFin.rndNum = rndNum
    y_train, X_train = rndFin.createTrainSet(fine_set)
    y_trainBin, train_wt = rndFin.createTrainWtYtrain(y_train)
    y_testCoarse, y_sampleWeight, X_test = rndFin.createTestSet(test_part, train_wt)
    rndFin.trainClassifier(X_train, y_trainBin)
    y_predCoarse, y_pred_score = rndFin.predictTestSet(X_test)
    rndFin.printConfMatrix(y_testCoarse, y_predCoarse, rnd_results_fine)
    rndFin.plotRocPrCurves(y_testCoarse, y_pred_score, y_sampleWeight, rnd_results_fine)

    ##### Append round time and fold counts
    m.appendRndTimesFoldCnts(rndNum, 'coarse', rnd_results_coarse, coarse_set, start_time)
    m.appendRndTimesFoldCnts(rndNum, 'fine', rnd_results_fine, fine_set, start_time)

    fileName = open('results/rnd_results_fine','wb')
    pickle.dump(rnd_results_fine,fileName)
    fileName.close()
    fileName = open('results/rnd_results_coarse','wb')
    pickle.dump(rnd_results_coarse,fileName)
    fileName.close()





