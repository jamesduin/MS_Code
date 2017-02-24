import numpy as np
import time
import methodsFFR as m
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
    os.chdir('_results')
    dataDir = '../../'
else:
    os.chdir('/work/scott/jamesd/runFFRR_Cst8')
    dataDir = '/home/scott/jamesd/MS_Code/'


FFR = float(sys.argv[1])
testFold = int(sys.argv[2])
batch = Decimal(100.0)
fineCost = Decimal(8.0)
coarseCost = Decimal(1.0)
add = dict()
add['fine'] = batch*(Decimal(FFR))/fineCost
add['coarse'] = batch*(Decimal(1.0)-Decimal(FFR))/coarseCost
roundSize = int(add['fine']+add['coarse']+Decimal(1))
results = []
m.addPrint(results,['batch']+[float(batch)]+['FFR']+[float(FFR)]+['fineCost']
+[float(fineCost)]+['coarseCost']+[float(coarseCost)]+['addFine']+[float(add['fine'])]
           +['addCoarse']+[float(add['coarse'])]+['roundSize']+[roundSize])
FFR = str(FFR).replace('.','p')

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
while((18088-instanceCount) > roundSize):
#while(rndNum < 2):
    start_time.append(time.perf_counter())
    if(rndNum>=1):
        ###### run confidence estimate for coarse and fine
        m.confEstAdd(results,classes_all,sets,rnds,add)
    rndNum += 1
    rnds = dict()
    rnds['coarse'] = m.CoarseRound(testFold, rndNum, FFR)
    rnds['fine'] = m.FineRound(testFold, rndNum, FFR)
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
    m.predictCombined(results,y_pred_score,y_testCoarse,y_sampleWeight,rndNum,testFold,FFR)
    ##### Append round time and fold counts
    instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, results, sets, start_time)
    m.appendSetTotal(rndNum, results, classes_all,'classes_all')
    tot = time.perf_counter() - start_time[0]
    m.addPrint(results,['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
        *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
    fileName = open('results/FFR_'+FFR+'_'+str(testFold)+'.res','wb')
    pickle.dump(results,fileName)
    fileName.close()


