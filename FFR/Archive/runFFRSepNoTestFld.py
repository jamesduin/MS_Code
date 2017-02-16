import numpy as np
import time
import methodsFFR as m
from sklearn import preprocessing
import pickle
import copy
import sys
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
print(rootDir)
if(rootDir == 'py'):
    dataDir = '../../'
    os.chdir('_results')
else:
    os.chdir('/work/scott/jamesd/resultsFFR_1')
    dataDir = '/home/scott/jamesd/MS_Code/'
FFR = float(sys.argv[1])
start_time = [time.perf_counter()]
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
sets = dict()
sets['coarse'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
sets['fine'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
strt_results = []
batch = 160
fineCost = 1
add = dict()
add['fine'] = int(batch*FFR/fineCost)
add['coarse'] = int(batch - add['fine'])
m.addPrint(strt_results,['batch']+[batch]+['FFR']+[FFR]+['fineCost']
+[fineCost]+['addFine']+[add['fine']]+['addCoarse']+[add['coarse']])
FFR = str(FFR).replace('.','_')
part_all = m.loadScaledPartData(dataDir)
m.printClsVsFolds(strt_results,part_all, 'all')
for i in sorted(part_all):
    for index in range(len(part_all[i])):
        classes_all[part_all[i][index][0]].append(part_all[i][index])
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


instanceCount = 0
rndNum = 0
while((18088-instanceCount) > 100):
#while(rndNum < 3):
    start_time.append(time.perf_counter())
    if(rndNum>1):
        ###### run confidence estimate for coarse and fine
        m.confEstAdd(classes_all,sets,rnds,add)
    rndNum += 1
    rnds = dict()
    rnds['coarse'] = m.CoarseRound( rndNum, FFR)
    rnds['fine'] = m.FineRound( rndNum, FFR)
    #### Run rounds
    for lvl in ['coarse', 'fine']:
        setParts = rnds[lvl].createSetParts(sets[lvl])
        for fld in range(1,11):
            y_train, X_train = rnds[lvl].createTrainSet(setParts, fld)
            y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train)
            y_testCoarse, y_sampleWeight, X_test = rnds[lvl].createTestSet(setParts[fld])
            rnds[lvl].trainClassifier(X_train, y_trainCoarse,fld)
            y_predCoarse, y_pred_score = rnds[lvl].predictTestSet(X_test)
            rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse, rnd_results[lvl],fld)
            rnds[lvl].plotRocPrCurves(y_testCoarse, y_pred_score, y_sampleWeight, rnd_results[lvl],fld)
    for lvl in ['coarse', 'fine']:
        ##### Append round time and fold counts
        instanceCount = m.appendRndTimesFoldCnts(FFR, rndNum, lvl, rnd_results[lvl], sets[lvl], start_time)
        tot = time.perf_counter() - start_time[0]
        m.addPrint(rnd_results[lvl],['Total Time:']+['{:.0f}hr {:.0f}m {:.2f}sec'.format(
            *divmod(divmod(tot,60)[0],60),divmod(tot,60)[1])])
        fileName = open('results/'+lvl+'_'+str(FFR)+'.res','wb')
        pickle.dump(rnd_results[lvl],fileName)
        fileName.close()


