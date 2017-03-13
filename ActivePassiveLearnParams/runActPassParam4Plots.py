import numpy as np
import time
import methodsActPassParam4Plots as m
from sklearn import preprocessing
import pickle
import copy
import sys
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
clfType = sys.argv[1]
dir = sys.argv[2]
rndType = sys.argv[3]
testFold = int(sys.argv[4])

baseDir = ''
if(rootDir == 'py'):
    baseDir = '../../'

else:
    os.chdir('/work/scott/jamesd/')
    baseDir = '/home/scott/jamesd/MS_Code/'


if(clfType == 'LogReg'):
    loadDir = baseDir+'data/partitionMinMaxScaled/partitionMinMaxScaled_'
elif(clfType == 'SVM'):
    loadDir = baseDir+'data/partitionStdSclSel/partitionStdSclSel_'

if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir + '/log')
    os.makedirs(dir + '/results')
    os.makedirs(dir + '/thresh')
os.chdir(dir)



start_time = [time.perf_counter()]
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
sets = dict()
sets['coarse'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
sets['fine'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
strt_results = []
train_part = m.loadScaledPartData(loadDir)

m.printClsVsFolds(strt_results,train_part, 'all')
test_part = train_part[testFold]
del train_part[testFold]
m.printClsVsFolds(strt_results,train_part, 'train')
classTestTot = m.printClsVsFolds(strt_results,{testFold:test_part}, 'test')
if(classTestTot[7] == 2):
    ## not enough of class 5 to have 2 in the test set
    m.switchClass5instance(test_part,train_part)
    m.printClsVsFolds(strt_results,train_part, 'train_mod')
    m.printClsVsFolds(strt_results,{testFold: test_part}, 'test_mod')
classes_all.clear()
classes_all = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
for i in sorted(train_part):
    for index in range(len(train_part[i])):
        classes_all[train_part[i][index][0]].append(train_part[i][index])

m.printClassTotals(strt_results,classes_all)

rnd_results = dict()
rnd_results['coarse'] = strt_results[:]
rnd_results['fine'] = strt_results[:]
rnd_results['fineTrainOnCrs'] = strt_results[:]

#### randomly add to starter sets
start = [952,10,11,16,11,10,10,10,10]
print(start)
print("Sum start => "+str(np.sum(start)))
for i in sorted(classes_all):
    for j in range(int(start[i])):
        inst = classes_all[i].pop()
        for lvl in ['coarse','fine']:
            sets[lvl][i].append(inst)

classes = dict()
for lvl in ['coarse','fine']:
    classes[lvl] = copy.deepcopy(classes_all)
instanceCount = 0
rndNum = 0
threshResults = {'coarse':[],'fine':[], 'fineTrainOnCrs':[]}
#while((18088-instanceCount) > 100):
while(rndNum < 40):
    start_time.append(time.perf_counter())
    if(rndNum>=1):
        for lvl in ['coarse', 'fine']:
            if(rndType == 'passive'):
                m.randAdd(classes[lvl],sets[lvl],100)
            elif(rndType == 'active'):
                ###### run confidence estimate for coarse and fine
                m.confEstAdd(rnd_results[lvl],classes[lvl],sets[lvl],rnds[lvl],lvl,100)
    rndNum += 1
    rnds = dict()
    rnds['coarse'] = m.CoarseRound(testFold, rndNum, rndType)
    rnds['fine'] = m.FineRound(testFold, rndNum, rndType)
    rnds['fineTrainOnCrs'] = m.FineRound(testFold, rndNum, rndType)
    sets['fineTrainOnCrs'] = sets['coarse']
    #### Run rounds
    y_predCoarse = dict()
    y_pred_score = dict()
    for lvl in ['coarse','fine','fineTrainOnCrs']:
        y_train, X_train = rnds[lvl].createTrainSet(sets[lvl])
        y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train,rnd_results[lvl])
        y_testCoarse, y_sampleWeight, X_test = rnds[lvl].createTestSet(test_part)
        rnds[lvl].trainClassifier(X_train, y_trainCoarse,clfType)
        y_predCoarse[lvl], y_pred_score[lvl] = rnds[lvl].predictTestSet(X_test)
        rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse[lvl], rnd_results[lvl])
        fpr, tpr, threshRoc = rnds[lvl].plotRocCurves(y_testCoarse,
                                                      y_pred_score[lvl],
                                                      y_sampleWeight, rnd_results[lvl])
        precision, recall, threshPr = rnds[lvl].plotPrCurves(y_testCoarse,
                            y_pred_score[lvl], y_sampleWeight, rnd_results[lvl])
        m.addPrint(rnd_results[lvl],'threshRoc {}'.format(len(threshRoc)))
        m.addPrint(rnd_results[lvl],'threshPr {}'.format(len(threshPr)))

        for tInd,thresh in enumerate(threshRoc):
            y_predCrsThresh, y_pred_scr = rnds[lvl].predictTestSetThreshold(thresh,
                                                y_pred_score[lvl])
            rnds[lvl].printConfMatrixThresh(y_testCoarse,y_predCrsThresh,threshResults[lvl],
                                            'flsPos',fpr[tInd],
                                            'truPos',tpr[tInd],
                                            'thresh',thresh)

        for tInd,thresh in enumerate(threshPr):
            y_predCrsThresh, y_pred_scr = rnds[lvl].predictTestSetThreshold(thresh,
                                                y_pred_score[lvl])
            rnds[lvl].printConfMatrixThresh(y_testCoarse,y_predCrsThresh,threshResults[lvl],
                                            'rec', recall[tInd],
                                            'prec', precision[tInd],
                                            'thresh', thresh)

    ##### Append round time and fold counts
    instanceCount = dict()
    for lvl in ['coarse','fine','fineTrainOnCrs']:
        instanceCount[lvl] = m.appendRndTimesFoldCnts(testFold, rndNum,lvl,rnd_results[lvl],
                                                      sets[lvl], start_time)
        m.appendSetTotal(rndNum, rnd_results[lvl], classes_all, 'classes_all')
        tot = time.perf_counter() - start_time[0]
        m.addPrint(rnd_results[lvl], ['Total Time:'] + ['{:.0f}hr {:.0f}m {:.2f}sec'.format(
            *divmod(divmod(tot, 60)[0], 60), divmod(tot, 60)[1])])
        f = open('results/'+rndType+'_'+lvl+'_'+str(testFold)+'.res','wb')
        pickle.dump(rnd_results[lvl], f)
        f.close()
    instanceCount = max([instanceCount['fine'], instanceCount['coarse']])
    #if(rndNum == 100 or (18088-instanceCount) <= 100 ):
    for lvl in ['coarse', 'fine','fineTrainOnCrs']:
        f = open('thresh/thresh_'+str(rndNum)+'_' + rndType + '_' + lvl + '_' + str(testFold) + '.res', 'wb')
        pickle.dump(threshResults[lvl], f)
        f.close()
        threshResults[lvl] = []

