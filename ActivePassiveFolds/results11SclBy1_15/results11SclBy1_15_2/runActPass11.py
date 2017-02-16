import numpy as np
import time
import methodsActPass11 as m
from sklearn import preprocessing
import pickle
import copy
import sys
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'py'):
    dataDir = '../'
else:
    os.chdir('/work/scott/jamesd/results11SclBy1_15_2')
    dataDir = '/home/scott/jamesd/MS_Code/'




rndType = sys.argv[1]
testFolds = sys.argv[2:]
for testFold in testFolds:
    testFold = int(testFold)
    start_time = [time.perf_counter()]
    classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    sets = dict()
    sets['coarse'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    sets['fine'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    strt_results = []
    train_part = m.loadScaledPartData(dataDir)
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
    while((18088-instanceCount) > 100):
    #while(rndNum < 10):
        start_time.append(time.perf_counter())
        if(rndNum>1):
            for lvl in ['coarse', 'fine']:
                if(rndType == 'passive'):
                    m.randAdd(classes[lvl],sets[lvl],100)
                elif(rndType == 'active'):
                    ###### run confidence estimate for coarse and fine
                    m.confEstAdd(classes[lvl],sets[lvl],rnds[lvl],100)
        rndNum += 1
        rnds = dict()
        rnds['coarse'] = m.CoarseRound(testFold, rndNum, rndType)
        rnds['fine'] = m.FineRound(testFold, rndNum, rndType)
        #### Run rounds
        for lvl in ['coarse', 'fine']:
            y_train, X_train = rnds[lvl].createTrainSet(sets[lvl])
            y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train)
            y_testCoarse, y_sampleWeight, X_test = rnds[lvl].createTestSet(test_part)
            rnds[lvl].trainClassifier(X_train, y_trainCoarse)
            y_predCoarse, y_pred_score = rnds[lvl].predictTestSet(X_test)
            rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse, rnd_results[lvl])
            rnds[lvl].plotRocPrCurves(y_testCoarse, y_pred_score, y_sampleWeight, rnd_results[lvl])
        for lvl in ['coarse', 'fine']:
            ##### Append round time and fold counts
            instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, lvl, rnd_results[lvl], sets[lvl], start_time)
            tot = time.perf_counter() - start_time[0]
            m.addPrint(rnd_results[lvl],['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
                *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
            fileName = open('results/'+rndType+'_'+lvl+'_'+str(testFold)+'.res','wb')
            pickle.dump(rnd_results[lvl],fileName)
            fileName.close()


