import numpy as np
import time
import methodsFFRParam as m
from sklearn import preprocessing
import pickle
import copy
import sys
import re
import os
from decimal import *
import numpy as np
import random
getcontext().prec = 8
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'py'):
    os.chdir('BanditTest')
    dataDir = '../../'
else:
    os.chdir('/work/scott/jamesd/BanditTest_Cst8')
    dataDir = '/home/scott/jamesd/MS_Code/'


testFold = int(sys.argv[1])
batch = Decimal(100.0)
fineCost = Decimal(8.0)
coarseCost = Decimal(1.0)


add = dict()
add['fine'] = batch/fineCost
add['coarse'] = batch/coarseCost
roundSize = 100
results = []
m.addPrint(results,['batch']+[float(batch)]+['fineCost']
+[float(fineCost)]+['coarseCost']+[float(coarseCost)]+['addFine']+[float(add['fine'])]
           +['addCoarse']+[float(add['coarse'])]+['roundSize']+[roundSize])


start_time = [time.perf_counter()]
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
sets = dict()
sets['coarse'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
sets['fine'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
train_part = m.loadScaledPartData(dataDir)
m.printClsVsFolds(results,train_part, 'all')
test_part = train_part[testFold]
del train_part[testFold]
m.printClsVsFolds(results,train_part, 'train')
classTestTot = m.printClsVsFolds(results,{testFold:test_part}, 'test')
if(classTestTot[7] == 2):
    ## not enough of class 5 to have 2 in the test set
    m.switchClass5instance(test_part,train_part)
    m.printClsVsFolds(results,train_part, 'train_mod')
    m.printClsVsFolds(results,{testFold: test_part}, 'test_mod')
classes_all.clear()
classes_all = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
for i in sorted(train_part):
    for index in range(len(train_part[i])):
        classes_all[train_part[i][index][0]].append(train_part[i][index])

m.printClassTotals(results,classes_all)



#### randomly add to starter sets
start = [952,10,11,16,11,10,10,10,10]
print(start)
print("Sum start => "+str(np.sum(start)))
for i in sorted(classes_all):
    for j in range(int(start[i])):
        inst = classes_all[i].pop()
        for lvl in ['coarse','fine']:
            sets[lvl][i].append(inst)

instanceCount = 0
rndNum = 0
combPredScoreRnd = dict()
rndReward = dict()
#while((18088-instanceCount) > roundSize and rndNum <= 750):
while(rndNum < 3):
    start_time.append(time.perf_counter())
    if(rndNum>=1):
        ###### run confidence estimate for coarse and fine
        m.confEstAdd(results,classes_all,sets,rnds,add)
    rndNum += 1
    rnds = dict()
    rnds['coarse'] = m.CoarseRound(testFold, rndNum)
    rnds['fine'] = m.FineRound(testFold, rndNum)
    y_testCoarse, y_sampleWeight, X_test = m.createTestSet(test_part)
    #### Run rounds
    y_predCoarse = dict()
    y_pred_score = dict()
    for lvl in ['coarse', 'fine']:
        y_train, X_train = rnds[lvl].createTrainSet(sets[lvl])
        y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train)
        rnds[lvl].trainClassifier(X_train, y_trainCoarse)
        y_predCoarse[lvl], y_pred_score[lvl] = rnds[lvl].predictTestSet(X_test)
        rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse[lvl], results)
        rnds[lvl].plotRocPrCurves(y_testCoarse, y_pred_score[lvl], y_sampleWeight, results)
    m.predictCombined(results,y_pred_score,y_testCoarse,y_sampleWeight,rndNum,testFold)
    combPredScoreRnd[rndNum] = m.predictCombUnlabeled()
    ##### Append round time and fold counts
    instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, results, sets, start_time)
    if(rndNum>=2):
        print(combPredScoreRnd[rndNum - 1][:20])
        print(combPredScoreRnd[rndNum][:20])
        dif = combPredScoreRnd[rndNum-1] - combPredScoreRnd[rndNum]
        print(dif[:20])
        absDif = np.abs(combPredScoreRnd[rndNum-1] - combPredScoreRnd[rndNum])
        print(absDif[:20])
        absDifLog = np.log(np.abs(combPredScoreRnd[rndNum-1] - combPredScoreRnd[rndNum]))
        print(absDifLog[:20])
        sumAbsDifLog = np.sum(np.log(np.abs(combPredScoreRnd[rndNum-1] - combPredScoreRnd[rndNum])))
        print(sumAbsDifLog)

    m.appendSetTotal(rndNum, results, classes_all,'classes_all')
    tot = time.perf_counter() - start_time[0]
    m.addPrint(results,['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
        *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
    fileName = open('results/Bandit_'+str(testFold)+'.res','wb')
    pickle.dump(results,fileName)
    fileName.close()


