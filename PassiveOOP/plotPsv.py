import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'Users' or rootDir == 'py'):
    dataDir = '../'
else:
    os.chdir('/work/scott/jamesd/')
    dataDir = '/home/scott/jamesd/MS_Code/'


def getRndTypeSet(resultsDir):
    rndTypeSet = set()
    for fname in glob.glob(resultsDir + '/*.res'):
        # print(fname)
        file = re.split("[/\.]", fname)[-2]
        # print(file)
        rndType = re.split("[_]", file)
        # print(rndType)
        print(rndType[0] + '_' + rndType[1])
        rndTypeSet.add(rndType[0] + '_' + rndType[1])
    print(rndTypeSet)
    return rndTypeSet

#resultsDir = 'runFFR_Cst16/results'
#resultsDir = 'runFFR_Cst1/results'
#resultsDir = 'resultsRBF11sclBy1_15/results'
#resultsDir = 'resultsSclBy1/results'
resultsDir = 'results'

rndTypeSet = getRndTypeSet(resultsDir)

rndTypeFoldMat = dict()
for type in rndTypeSet:
    foldMatrix = dict()
    for fold in range(1,11):
        for fname in glob.glob(resultsDir+'/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0]+'_'+rndType[1]
            #if (type == instType and str(fold) == rndType[2] ):
            print(fold)
            print(instType)
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

plt.figure()
#with plt.style.context('fivethirtyeight'):
plt.style.use('ggplot')
for type in rndTypeSet:
    prs = []
    for fold in sorted(rndTypeFoldMat[type]):
        prFold = []
        for rec in rndTypeFoldMat[type][fold]:
            #if(not isinstance(rec,str)):
            if('pr'in rec):
                ind =[i for i in range(len(rec)) if rec[i] == 'pr']
                prFold.append(rec[ind[0]+1])
        prFold = np.array(prFold).reshape(len(prFold),1)
        if prs == []:
            prs =prFold
        else:
            prs = np.hstack((prs,prFold))
    x_pr = np.array(range(1,len(prs)+1))
    y_pr = np.mean(prs, axis=1)
    plt.plot(x_pr,y_pr, label = 'avg_'+type)

plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title(resultsDir)
plt.legend(loc="lower right")
plt.savefig(resultsDir+'/Passive.png')

