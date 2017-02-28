import numpy as np
import time
import methodsPsv as m
from sklearn import preprocessing
import pickle
import copy
import sys
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'py'):
    os.chdir('FindThresholdSVM')
    dataDir = '../../'
else:
    os.chdir('/work/scott/jamesd/results')
    dataDir = '/home/scott/jamesd/MS_Code/'



testFold = int(sys.argv[1])
clfType = 'SVM'  #LogReg, SVM
results = []

start_time = [time.perf_counter()]
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
train_part = m.loadScaledPartData(clfType)
#train_part = m.loadAndProcessData('classes',clfType)  #classes, classes_subset
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




instanceCount = 0
rndNum = 1
start_time.append(time.perf_counter())
rnds = dict()
rnds['coarse'] = m.CoarseRound(testFold, rndNum, "Psv")
rnds['fine'] = m.FineRound(testFold, rndNum, "Psv")
y_testCoarse, y_sampleWeight, X_test, y_test = m.createTestSet(test_part)
#### Run rounds
y_predCoarse = dict()
y_pred_score = dict()
for lvl in ['coarse', 'fine']:
#for lvl in ['fine']:
    y_train, X_train = rnds[lvl].createTrainSet(classes_all)
    y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train)
    rnds[lvl].trainClassifier(X_train, y_trainCoarse,clfType)
    y_predCoarse[lvl], y_pred_score[lvl] = rnds[lvl].predictTestSet(X_test)
    rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse[lvl], results)
    ###### log the errors
    # err_file = open('jaccard/'+clfType+'_'+lvl+'_'+str(testFold)+'.txt', 'w')
    # for i,pred in enumerate(y_predCoarse[lvl]):
    #     if(y_predCoarse[lvl][i] != y_testCoarse[i]):
    #         m.printDataInstance(np.array([y_test[i]]+list(X_test[i])), err_file)
    # err_file.close()
    fpr, tpr, threshRoc = rnds[lvl].plotRocCurves(y_testCoarse, y_pred_score[lvl], y_sampleWeight, results)
    precision, recall, threshPr = rnds[lvl].plotPrCurves(y_testCoarse, y_pred_score[lvl], y_sampleWeight, results)
    print('threshRoc {}'.format(len(threshRoc)))
    print('threshPr {}'.format(len(threshPr)))
    for tInd,thresh in enumerate(threshRoc):
        y_predCrsThresh, y_pred_scr = rnds[lvl].predictTestSetThreshold(thresh,y_pred_score[lvl])
        rnds[lvl].printConfMatrixThresh(y_testCoarse,y_predCrsThresh,results,
                                        'flsPos',fpr[tInd],
                                        'truPos',tpr[tInd],
                                        'thresh',thresh)

    for tInd,thresh in enumerate(threshPr):
        y_predCrsThresh, y_pred_scr = rnds[lvl].predictTestSetThreshold(thresh,y_pred_score[lvl])
        rnds[lvl].printConfMatrixThresh(y_testCoarse,y_predCrsThresh,results,
                                        'rec', recall[tInd],
                                        'prec', precision[tInd],
                                        'thresh', thresh)

#m.predictCombined(results,y_pred_score,y_testCoarse,y_sampleWeight,rndNum,testFold,"Psv")
##### Append round time and fold counts
instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, results, classes_all, start_time)
m.appendSetTotal(rndNum, results, classes_all,'classes_all',testFold)
tot = time.perf_counter() - start_time[0]
m.addPrint(results,['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
    *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
fileName = open('results/Psv_'+str(testFold)+'.res','wb')
pickle.dump(results,fileName)
fileName.close()

