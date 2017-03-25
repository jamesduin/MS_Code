import numpy as np
import time
import methodsFFRBandit as m
from math import isinf
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



batch = Decimal(100.0)
fineCost = Decimal(sys.argv[2])
coarseCost = Decimal(1.0)
testFold = int(sys.argv[3])
dir = sys.argv[1] +'_'+ str(fineCost).replace('.','p')

baseDir = ''
if(rootDir == 'py'):
    dataDir = '../../'

else:
    os.chdir('/work/scott/jamesd/')
    dataDir = '/home/scott/jamesd/MS_Code/'

if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir + '/log')
    os.makedirs(dir + '/results')
os.chdir(dir)




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
crsFinePredScore = dict()
rndClassifier = dict()
rndGain = []
armStay = [0.0]
armSwitch = [0.0]
rndP = ['shift','start','pCrs']
armPlayed = ['start']
addCur = dict()
while((18088-instanceCount) > roundSize and rndNum <= 750):
#while(rndNum < 3):
    start_time.append(time.perf_counter())
    rndNum += 1
    if(rndNum>=2):
        ###### run confidence estimate for coarse and fine
        if(rndP[rndNum] == 'pCrs'):
            addCur['coarse'] = add['coarse']
            addCur['fine'] = Decimal(0.0)

        elif(rndP[rndNum] == 'pFin'):
            addCur['coarse'] = Decimal(0.0)
            addCur['fine'] = add['fine']
        # addCur['coarse'] = add['coarse']
        # addCur['fine'] = Decimal(0.0)
        print(addCur)
        m.confEstAdd(results,classes_all,sets,rnds,addCur)
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
    rndClassifier[rndNum] = rnds
    ##### Append round time and fold counts
    instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, results, sets,classes_all, start_time)
    if(rndNum>=2):
        crsFinePredScore[rndNum-1] = m.getScoresUnlabeledX(results,classes_all,rndClassifier[rndNum-1])
        combPredScoreRnd[rndNum-1] = m.predictCombinedUnlabeled(results, crsFinePredScore[rndNum-1],rndNum-1)
        crsFinePredScore[rndNum] = m.getScoresUnlabeledX(results, classes_all, rndClassifier[rndNum])
        combPredScoreRnd[rndNum] = m.predictCombinedUnlabeled(results, crsFinePredScore[rndNum], rndNum)

        # print(combPredScoreRnd[rndNum - 1][:20])
        # print(combPredScoreRnd[rndNum][:20])
        dif = combPredScoreRnd[rndNum-1] - combPredScoreRnd[rndNum]
        # print(dif[:20])
        m.addPrint(results,'length dif all: {}'.format(len(dif)))
        absDif = np.abs(combPredScoreRnd[rndNum-1] - combPredScoreRnd[rndNum])
        absDif = absDif[np.nonzero(absDif)]
        m.addPrint(results,'length dif no zeros: {}'.format(len(absDif)))
        # print(absDif[:20])
        # absDifLog = np.log(absDif)
        # print(absDifLog[:20])
        sumAbsDifLog = np.sum(np.log(absDif))
        m.addPrint(results,'sumAbsDifLog: {}'.format(sumAbsDifLog))
        normX = np.linalg.norm(m.getUnlabeledX(results,classes_all))
        m.addPrint(results, 'normX: {}'.format(normX))
        gain = sumAbsDifLog/normX
        rndGain.append([rndNum,gain])
        m.addPrint(results, 'rndGain: {}'.format(rndGain))
        m.addPrint(results, 'rndP: {}'.format(rndP))

        armStay.append(0.0)
        if (rndP[rndNum-1] == 'pCrs' and rndP[rndNum]=='pFin'):
            armSwitch.append(-gain/np.abs(gain))
        elif(rndP[rndNum-1]=='pFin' and rndP[rndNum]=='pCrs'):
            armSwitch.append(gain/np.abs(gain))
        else:
            armSwitch.append(0.0)
        m.addPrint(results, 'armStay: {}'.format(armStay))
        m.addPrint(results, 'armSwitch: {}'.format(armSwitch))

        armStayAvgRew = np.mean(np.array(armStay))
        armSwitchAvgRew = np.mean(np.array(armSwitch))
        m.addPrint(results, 'armStayAvgRew: {}'.format(armStayAvgRew))
        m.addPrint(results, 'armSwitchAvgRew: {}'.format(armSwitchAvgRew))

        epsilon = min(1.0, 2.0/rndNum)
        selectBestChance = 1- epsilon
        rand = random.random()
        playBest = False
        if(rand <= selectBestChance):
            playBest = True
        if(playBest):
            if(armSwitchAvgRew > armStayAvgRew):
                armPlayed.append('{}-{}-{:.3f}'.format('playBest','armSwitch',selectBestChance))
                if(rndP[rndNum] == 'pFin'):
                    rndP.append('pCrs')
                elif(rndP[rndNum] == 'pCrs'):
                    rndP.append('pFin')
            elif (armStayAvgRew > armSwitchAvgRew):
                armPlayed.append('{}-{}-{:.3f}'.format('playBest', 'armStay',selectBestChance))
                if (rndP[rndNum] == 'pFin'):
                    rndP.append('pFin')
                elif (rndP[rndNum] == 'pCrs'):
                    rndP.append('pCrs')
            elif (armStayAvgRew == armSwitchAvgRew):
                randArm = random.random()
                if (randArm < 0.5):
                    armPlayed.append('{}-{}-{:.3f}'.format('playBestEqual', 'armSwitch', selectBestChance))
                    if (rndP[rndNum] == 'pFin'):
                        rndP.append('pCrs')
                    elif (rndP[rndNum] == 'pCrs'):
                        rndP.append('pFin')
                else:
                    armPlayed.append('{}-{}-{:.3f}'.format('playBestEqual', 'armStay', selectBestChance))
                    if (rndP[rndNum] == 'pFin'):
                        rndP.append('pFin')
                    elif (rndP[rndNum] == 'pCrs'):
                        rndP.append('pCrs')
        else:
            randArm = random.random()
            if(randArm < 0.5):
                armPlayed.append('{}-{}-{:.3f}'.format('randArm', 'armSwitch',selectBestChance))
                if (rndP[rndNum] == 'pFin'):
                    rndP.append('pCrs')
                elif (rndP[rndNum] == 'pCrs'):
                    rndP.append('pFin')
            else:
                armPlayed.append('{}-{}-{:.3f}'.format('randArm', 'armStay',selectBestChance))
                if (rndP[rndNum] == 'pFin'):
                    rndP.append('pFin')
                elif (rndP[rndNum] == 'pCrs'):
                    rndP.append('pCrs')
        m.addPrint(results, 'armPlayed: {}'.format(armPlayed))
    #m.appendSetTotal(rndNum, results, classes_all,'classes_all')
    tot = time.perf_counter() - start_time[0]
    m.addPrint(results,['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
        *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
    fileName = open('results/Bandit_BpT_'+str(testFold)+'.res','wb')
    pickle.dump(results,fileName)
    fileName.close()


