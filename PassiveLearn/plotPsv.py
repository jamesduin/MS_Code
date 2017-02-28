import numpy as np
import matplotlib
matplotlib.use('Agg')
import pprint as pp
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'py'):
    os.chdir('FindThresholdLogReg')  #FmtResultsSVM, FmtResultsLogReg
    dataDir = '../../'


def getRndTypeSet(resultsDir):
    rndTypeSet = set()
    for fname in glob.glob(resultsDir + '/*.res'):
        file = re.split("[/\.]", fname)[-2]
        rndType = re.split("[_]", file)
        #print(rndType[0])
        rndTypeSet.add(rndType[0])
    #print(rndTypeSet)
    return rndTypeSet

resultsDir = 'results'

rndTypeSet = getRndTypeSet(resultsDir)

rndTypeFoldMat = dict()
for type in rndTypeSet:
    foldMatrix = dict()
    for fold in range(1,11):
        for fname in glob.glob(resultsDir+'/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0]
            if (type == instType and str(fold) == rndType[1] ):
                #print(fold)
                #print(instType)
                foldMatrix[fold] = []
                results = []
                try:
                    results = pickle.load(open(fname, 'rb'))
                except EOFError:
                    pass
                for result in results:
                    foldMatrix[fold].append(result)
    rndTypeFoldMat[type] = foldMatrix

max = []
for fold in foldMatrix:
    max.append(len(foldMatrix[fold]))
#print(np.max(max))

for type in rndTypeFoldMat:
    ### print out the folds
    f = open(resultsDir+'/_' + type + '.txt', 'w')
    for fold in rndTypeFoldMat[type]:
        for rec in rndTypeFoldMat[type][fold]:
            f.write(str(rec)+'\n')
    f.close()


resultMat = []
colNum = 0
for fnd in ['pr','roc','acc','f1']:
#for fnd in ['tn','fn','fp','tp']:
    for type in rndTypeSet:
        outFind = dict()
        for lvl in ['coarse','fine']:
            resultMat.append([])
            print('{} {}'.format(colNum,fnd))
            resultMat[colNum].append(lvl+'-'+fnd)
            prFold = []
            for fold in sorted(rndTypeFoldMat[type]):
                for rec in rndTypeFoldMat[type][fold]:
                    #if(not isinstance(rec,str)):
                    if(fnd in rec and lvl in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        prFold.append(rec[ind[0]+1])
                        #resultMat[colNum].append('{:.3f}'.format(rec[ind[0]+1]))
                        resultMat[colNum].append('{:}'.format(rec[ind[0] + 1]))
            prFold = np.array(prFold)
            print(prFold)
            #resultMat[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
            resultMat[colNum].append('avg {:.1f}'.format(np.mean(prFold)))
            colNum += 1


        f = open('output2.txt', 'w')
        for i in range(len(resultMat)):
            f.write('|l|')
        f.write('\n')
        for i,row in enumerate(resultMat[0]):
            for j,col in enumerate(resultMat[:-1]):
                f.write('{} & '.format(resultMat[j][i]))
            f.write('{} \\\\'.format(resultMat[-1][i]))
            f.write('\n')
        f.close()
