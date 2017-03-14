import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os


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



# loadDir = 'runActPass4plots/results'
# resultsDir = 'runActPass4plots/results'
loadDir = 'runActPassLogReg4plots/results'
resultsDir = 'runActPassLogReg4plots/'
#resultsDir = 'runActPassSVMTimeOut/results'
#resultsDir = '../ThesisWriteUp/fig/'
# loadDir = 'methodsActPassParam4PlotsRocAll/results'
# resultsDir = 'methodsActPassParam4PlotsRocAll/results'
# loadDir = 'methodsActPassParam4PlotsRocAllNoSW/results'
# resultsDir = 'methodsActPassParam4PlotsRocAllNoSW/results'

name = re.split("[/]", loadDir)[0]
rndTypeSet = getRndTypeSet(loadDir)

rndTypeFoldMat = dict()
for type in rndTypeSet:
    foldMatrix = dict()
    for fold in range(1,11):
        for fname in glob.glob(loadDir+'/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0]+'_'+rndType[1]
            if (type == instType and str(fold) == rndType[2] ):
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
    f = open(loadDir+'/_' + type + '.txt', 'w')
    for fold in rndTypeFoldMat[type]:
        for rec in rndTypeFoldMat[type][fold]:
            f.write(str(rec)+'\n')
    f.close()



cVals = [(0.90000000000000002, 0.25162433333706963, 0.12708553664078234, 1.0),
(0.90000000000000002, 0.48317993787976221, 0.25162433333706957, 1.0),
(0.86294117647058821, 0.67619870060118625, 0.3711206857265133, 1.0),
(0.69352941176470584, 0.81992038432147962, 0.48784802082520451, 1.0),
(0.53117647058823525, 0.89098219195263628, 0.58975546501210829, 1.0),
(0.36882352941176472, 0.89098219195263628, 0.67984446124709452, 1.0),
(0.2064705882352941, 0.8199203843214794, 0.75630966447410342, 1.0),
(0.037058823529411734, 0.67619870060118592, 0.81992038432147951, 1.0),
(0.12529411764705883, 0.48317993787976199, 0.86410948083716543, 1.0),
(0.28764705882352942, 0.25162433333706941, 0.89098219195263628, 1.0),
(0.45000000000000001, 0.0, 0.90000000000000002, 1.0)]

rndTypeSet = ['active_fine','passive_fine',
              'active_coarse','passive_coarse']
# rndTypeSet = ['active_fine','active_coarse',
#               'active_fineTrainOnCrs','active_coarseTrainOnFin']
colInd = {'active_fine':0,'passive_fine':4,
              'active_coarse':7,'passive_coarse':10}
# colInd = {'active_fine':0,'active_coarse':4,
#               'active_fineTrainOnCrs':7,'active_coarseTrainOnFin':10}
yLabel = {'pr':'PR-AUC','roc':'ROC-AUC',
              'f1':'F-measure','acc':'Accuracy'}

findList = ['pr','roc','f1','acc']
for find in findList:
    plt.figure()
    # with plt.style.context('fivethirtyeight'):
    plt.style.use('ggplot')
    for type in rndTypeSet:
        prs = []
        for fold in sorted(rndTypeFoldMat[type]):
            prFold = []
            for rec in rndTypeFoldMat[type][fold]:
                #if(not isinstance(rec,str)):
                if(find in rec):
                    ind =[i for i in range(len(rec)) if rec[i] == find]
                    prFold.append(rec[ind[0]+1])
            prFold = np.array(prFold).reshape(len(prFold),1)
            if len(prs) == 0:
                prs =prFold
            else:
                minRL = min(len(prs), len(prFold))
                prs = np.hstack((prs[:minRL], prFold[:minRL]))
        x_pr = np.array(range(1,len(prs)+1))
        y_pr = np.mean(prs, axis=1)

        plt.plot(x_pr,y_pr, label = type, linewidth=2.0,
                 fillstyle='none',color=cVals[colInd[type]])

    plt.ylabel(yLabel[find])
    plt.xlabel('Iteration')
    plt.title('Active vs. Passive Learning')
    plt.legend(loc="lower right")
    plt.savefig(resultsDir+name+'_'+find+'.png')
    plt.close()

