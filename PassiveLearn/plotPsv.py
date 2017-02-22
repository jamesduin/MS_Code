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
        file = re.split("[/\.]", fname)[-2]
        rndType = re.split("[_]", file)
        print(rndType[0])
        rndTypeSet.add(rndType[0])
    print(rndTypeSet)
    return rndTypeSet

resultsDir = 'results/results'

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

# plt.figure()
# #with plt.style.context('fivethirtyeight'):
# plt.style.use('ggplot')
for fnd in ['pr','roc','acc','f1']:
    for type in rndTypeSet:
        prs = dict()
        for lvl in ['coarse','fine']:
            prs[lvl]=[]
            for fold in sorted(rndTypeFoldMat[type]):
                prFold = []
                for rec in rndTypeFoldMat[type][fold]:
                    #if(not isinstance(rec,str)):
                    if(fnd in rec and lvl in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        prFold.append(rec[ind[0]+1])
                prFold = np.array(prFold).reshape(len(prFold),1)
                if prs[lvl] == []:
                    prs[lvl] =prFold
                else:
                    prs[lvl] = np.hstack((prs[lvl],prFold))
            print('{},{}'.format(lvl,fnd))
            for i in prs[lvl][0]:
                print('{:.3f}'.format(i))
            print('{},avg,{:.3f}'.format(lvl,np.mean(prs[lvl], axis=1)[0]))
#     plt.plot(x_pr,y_pr, label = 'avg_'+type)
#
# plt.ylabel('PR-AUC')
# plt.xlabel('Iteration')
# plt.title(resultsDir)
# plt.legend(loc="lower right")
# plt.savefig(resultsDir+'/FFR_PR.png')

