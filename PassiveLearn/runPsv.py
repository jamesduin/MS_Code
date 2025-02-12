import numpy as np
import time
import methodsPsv as m
from sklearn import preprocessing
import pickle
import copy
import sys
import re
import os
import shutil


testFold = int(sys.argv[1])
#dir = 'CostGamma/SVM_OrigWithFtune'
#dir = 'CostGamma/SVM_Cp1'
#dir = 'CostGamma/SVM_C1_Gp0029674'
#dir = 'CostGamma/SVM_C2_Gp0029674'
#dir = 'CostGamma/SVM_Cp1_Gp0029674'
#dir = 'CostGamma/SVM_Cp05_Gp0029674'
#dir = 'CostGamma/SVM_Cp15_Gp0029674'
#dir = 'CostGamma/SVM_Cp2_Gp0029674'
#dir = 'CostGamma/SVM_C1_Gp002'
#dir = 'CostGamma/SVM_C1_Gp001'
#dir = 'CostGamma/SVM_C1_Gp0005'
#dir = 'CostGamma/SVM_C1_Gp0015'
#dir = 'CostGamma/SVM_C1_Gp0025'
#dir = 'CostGamma/SVM_C1_Gp0035'
#dir = 'CostGamma/SVM_Cp1_Gp003'
#dir = 'CostGamma/SVM_Cp1_Gp0025'
#dir = 'CostGamma/SVM_Cp1_Gp002'
#dir = 'CostGamma/SVM_Cp1_Gp001'
#dir = 'CostGamma/SVM_Cp05_Gp002'
#dir = 'CostGamma/SVM_Cp15_Gp002'
#dir = 'CostGamma/SVM_Cp15_Gp001'
#dir = 'CostGamma/SVM_Cp5_Gp002'
#dir = 'CostGamma/SVM_Cp3_Gp002'
#dir = 'CostGamma/SVM_Cp15_Gp002_tolp00001'
#dir = 'CostGamma/SVM_C1_Gp0025_tolp00001'
#dir = 'ClassWeight/SVM_All'
#dir = 'ClassWeight/SVM_AllQuick'
#dir = 'RocPR/LogRegSampleWeight'
#dir = 'RocPR/LogRegNoSW'
#dir = 'RocPR/LogRegOrig'
#dir = 'Tolerance/LogReg_00001Redo'
#dir = 'RocPR/LogReg_NoSW'
#dir = 'RocPR/LogReg_NoSW_DropFalse'
#dir = 'RocPR/LogReg_DropFalse'
dir = 'Kernel/RbfOrig'
#dir = 'Kernel/Linear'
#dir = 'Kernel/polyDeg3'
#dir = 'Kernel/polyDeg6'
#dir = 'Kernel/sigmoid'
#fineCls = 8

if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir+'/coarse_results')
    os.makedirs(dir+'/fine_results')
    os.makedirs(dir+'/results')
os.chdir(dir)

results = []
start_time = [time.perf_counter()]
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
#loadDir =  '../../../data/partitionMinMaxScaled/partitionMinMaxScaled_'
#loadDir = '../../../data/partitionStdSclSel/partitionStdSclSel_'
#loadDir = '../../../data/partition_subset/partition_sub'
#loadDir = '../../../data/part_subStd/part_subStd_'

loadDir = '../../../data/part_subMinMax/part_subMinMax_'

#loadDir = '../../../data/part_subNorm/part_subNorm_'
#loadDir = '../../../data/part_subSel25/part_subSel25_'
#loadDir = '../../../data/part_subSel50/part_subSel50_'

#loadDir = '../../../data/part_subSel75/part_subSel75_'

#loadDir = '../../../data/part_subMinSel25/part_subMinSel25_'
#loadDir = '../../../data/part_subMinSel50/part_subMinSel50_'
#loadDir = '../../../data/part_subMinSel75/part_subMinSel75_'
train_part = m.loadScaledPartData(loadDir)


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
    y_train, X_train = rnds[lvl].createTrainSet(classes_all)
    y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train, results)
    rnds[lvl].trainClassifier(X_train, y_trainCoarse)
    y_predCoarse[lvl], y_pred_score[lvl] = rnds[lvl].predictTestSet(X_test)
    rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse[lvl], results)
    # rnds[lvl].predictTestSetFineCls(X_train,y_train,fineCls,'trainCls',results)
    # rnds[lvl].predictTestSetFineCls(X_test,y_test,fineCls,'testCls',results)



    fpr, tpr, threshRoc = rnds[lvl].plotRocCurves(y_testCoarse,
                                                  y_pred_score[lvl],
                                                  y_sampleWeight, results)
    precision, recall, threshPr = rnds[lvl].plotPrCurves(y_testCoarse,
                        y_pred_score[lvl], y_sampleWeight, results)
    m.addPrint(results,'y_testCoarse {}'.format(len(y_testCoarse)))
    m.addPrint(results, 'y_pred_score[] {}'.format(len(y_pred_score[lvl])))
    m.addPrint(results,'threshRoc {}'.format(len(threshRoc)))
    m.addPrint(results,'threshPr {}'.format(len(threshPr)))
    for tInd,thresh in enumerate(threshRoc):
        y_predCrsThresh, y_pred_scr = rnds[lvl].predictTestSetThreshold(thresh,
                                            y_pred_score[lvl])
        rnds[lvl].printConfMatrixThresh(y_testCoarse,y_predCrsThresh,results,
                                        'flsPos',fpr[tInd],
                                        'truPos',tpr[tInd],
                                        'thresh',thresh)

    for tInd,thresh in enumerate(threshPr):
        y_predCrsThresh, y_pred_scr = rnds[lvl].predictTestSetThreshold(thresh,
                                            y_pred_score[lvl])
        rnds[lvl].printConfMatrixThresh(y_testCoarse,y_predCrsThresh,results,
                                        'rec', recall[tInd],
                                        'prec', precision[tInd],
                                        'thresh', thresh)

#m.predictCombined(results,y_pred_score,y_testCoarse,y_sampleWeight,rndNum,testFold,"Psv")
##### Append round time and fold counts
instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, results, classes_all,
                                 start_time)
m.appendSetTotal(rndNum, results, classes_all,'classes_all',testFold)
tot = time.perf_counter() - start_time[0]
m.addPrint(results,['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
    *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
fileName = open('results/Psv_'+str(testFold)+'.res','wb')
pickle.dump(results,fileName)
fileName.close()

