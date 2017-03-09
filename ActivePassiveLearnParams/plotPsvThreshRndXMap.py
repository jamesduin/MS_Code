import numpy as np
import matplotlib
matplotlib.use('Agg')
import pprint as pp
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
from sklearn.metrics import auc
#os.chdir('FindThresholdSVM')  # FindThresholdSVM, FindThresholdLogReg
#os.chdir('FINAL/LogReg_All')
#os.chdir('FINAL/LogReg_All')
os.chdir('runActPassLogReg')

clftype = 'LogReg' #LogReg, SVM
#rndNums = [20,30,40,50,60]
rndNums = [70,60,50,40,30,20,10]


#type = 'active_fine'
#rndTypes = ['active_fine']#,'active_coarse']
rndTypes = ['active_coarse']
logLoc = 'thresh_100_'

lineSty = [[8, 1], [4, 1], [2, 1],
           [8, 1], [4, 1], [2, 1],
           [8, 1], [4, 1], [2, 1], [4, 1],
           [8, 1]]
markSty = ['s', '8', '>',
           's', '8', '>',
           's', '8', '>', '8',
           's']
markEvSty = [(1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8)]

cVals = [(0.90000000000000002, 0.25162433333706963, 0.12708553664078234, 1.0),
         (0.90000000000000002, 0.48317993787976221, 0.25162433333706957, 1.0),
         (0.86294117647058821, 0.67619870060118625, 0.3711206857265133, 1.0),
         #(0.69352941176470584, 0.81992038432147962, 0.48784802082520451, 1.0),
         (0.53117647058823525, 0.89098219195263628, 0.58975546501210829, 1.0),
         #(0.36882352941176472, 0.89098219195263628, 0.67984446124709452, 1.0),
         (0.2064705882352941, 0.8199203843214794, 0.75630966447410342, 1.0),
         #(0.037058823529411734, 0.67619870060118592, 0.81992038432147951, 1.0),
         (0.12529411764705883, 0.48317993787976199, 0.86410948083716543, 1.0),
         #(0.28764705882352942, 0.25162433333706941, 0.89098219195263628, 1.0),
         (0.45000000000000001, 0.0, 0.90000000000000002, 1.0)
         ]


def mapToAxisROC(x_map,resultMat):
    prFold = []
    for i, res in enumerate(x_map):
        prev = 0
        cur = 0
        for j, old in enumerate(resultMat[0]):
            cur = j
            if (res <= old):
                break
            prev = j
        y = np.array([resultMat[1][prev], resultMat[1][cur]])
        x = np.array([resultMat[0][prev], resultMat[0][cur]])
        if((x[0] - x[1]) != 0):
            m = (y[0] - y[1]) / (x[0] - x[1])
        else:
            m = 0.0
        b = y[1] - m * x[1]
        pr_inter = m * res + b
        # print('x_axis {}'.format(res))
        # print('pr_inter {}'.format(pr_inter))
        # print('point1 {}, {}'.format(resultMat[0][prev], resultMat[1][prev]))
        # print('point2 {}, {}'.format(resultMat[0][cur], resultMat[1][cur]))
        prFold.append(pr_inter)

    prFold = np.array(prFold).reshape(len(prFold), 1)
    return prFold

def mapToAxisPR(x_map,resultMat):
    prFold = []
    for i, res in enumerate(x_map):
        prev = 0
        cur = 0
        for j, old in enumerate(resultMat[0]):
            cur = j
            if (res >= old):
                break
            prev = j
        y = np.array([resultMat[1][prev], resultMat[1][cur]])
        x = np.array([resultMat[0][prev], resultMat[0][cur]])
        if((x[0] - x[1]) != 0):
            m = (y[0] - y[1]) / (x[0] - x[1])
        else:
            m = 0.0
        b = y[1] - m * x[1]
        pr_inter = m * res + b
        # print('x_axis {}'.format(res))
        # print('pr_inter {}'.format(pr_inter))
        # print('point1 {}, {}'.format(resultMat[0][prev], resultMat[1][prev]))
        # print('point2 {}, {}'.format(resultMat[0][cur], resultMat[1][cur]))
        prFold.append(pr_inter)

    prFold = np.array(prFold).reshape(len(prFold), 1)
    return prFold



def getFoldMatrix(type,rndNum):
    resultsDir = 'results'
    rndTypeFoldMat = dict()
    foldMatrix = dict()
    for fold in range(1, 11):
        for fname in glob.glob(resultsDir + '/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0] + '_' + rndType[1]
            if (type == instType and str(fold) == rndType[2]):
                foldMatrix[fold] = []
                results = []
                try:
                    results = pickle.load(open(fname, 'rb'))
                except EOFError:
                    pass
                for result in results:
                    foldMatrix[fold].append(result)
    rndTypeFoldMat[type] = foldMatrix

    AllRes = []
    colNum = 0
    finds = ['pr', 'roc', 'acc', 'f1']
    for fnd in finds:
        outFind = dict()
        lvl = re.split("[_]", type)[1]
        AllRes.append([])
        print(colNum)
        AllRes[colNum].append(lvl + '-' + fnd)
        prFold = []
        for fold in sorted(rndTypeFoldMat[type]):
            for rec in rndTypeFoldMat[type][fold]:
                # if(not isinstance(rec,str)):
                if ('rnd' in rec):
                    if (rec[1] == rndNum):
                        if (fnd in rec and lvl in rec):
                            ind = [i for i in range(len(rec)) if rec[i] == fnd]
                            prFold.append(rec[ind[0] + 1])
                            AllRes[colNum].append('{:.3f}'.format(rec[ind[0] + 1]))
        prFold = np.array(prFold)
        AllRes[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
        colNum += 1

    print(AllRes)
    resultsDir = 'thresh'
    type = logLoc + type
    rndTypeFoldMat = dict()
    foldMatrix = dict()
    for fold in range(1, 11):
        for fname in glob.glob(resultsDir + '/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0] + '_' + rndType[1] + '_' + rndType[2] + '_' + rndType[3]
            if (type == instType and str(fold) == rndType[4]):
                foldMatrix[fold] = []
                results = []
                try:
                    results = pickle.load(open(fname, 'rb'))
                except EOFError:
                    pass
                for result in results:
                    foldMatrix[fold].append(result)
    rndTypeFoldMat[type] = foldMatrix
    return AllRes,rndTypeFoldMat



for type in rndTypes:
    AllResRnds = dict()
    rndTypeFoldMatRnds = dict()
    for rndNum in rndNums:
        AllResRnds[rndNum], rndTypeFoldMatRnds[rndNum] = getFoldMatrix(type, rndNum)

    typeOrig = type
    type = logLoc + type
    pltRoc = plt
    pltRoc.style.use('ggplot')
    pltRoc.figure()
    rndCnt = -1
    for rndNum in rndNums:
        rndCnt += 1
        AllRes = AllResRnds[rndNum]
        rndTypeFoldMat = rndTypeFoldMatRnds[rndNum]

        prs = []
        x_map = np.arange(0.0, 1.001, 0.001)
        x_map = x_map.reshape(len(x_map),1)
        for fold in sorted(rndTypeFoldMat[type]):
        #for fold in [1,2]:
            lvl = re.split("[_]", type)[3]
            resultMat = []
            colNum = 0
            finds = ['flsPos', 'truPos']
            for fnd in finds:
                resultMat.append([])
                print('{} {}'.format(colNum,fnd))
                axisValFlds = []
                axisVal = []
                for rec in rndTypeFoldMat[type][fold]:
                    if('rnd' in rec):
                        if(rec[1] == rndNum):
                            if(lvl in rec and finds[0] in rec and finds[1] in rec ):#and finds[2] in rec):
                                ind =[i for i in range(len(rec)) if rec[i] == fnd]
                                axisVal.append(rec[ind[0]+1])
                                #resultMat[colNum].append('{:}'.format(rec[ind[0] + 1]))
                axisVal = np.array(axisVal).reshape(len(axisVal),1)
                print('axisVal: {} fld: {}'.format(len(axisVal),fold))
                resultMat[colNum] = axisVal
                colNum += 1

            prFold = mapToAxisROC(x_map, resultMat)
            if len(prs) == 0:
                prs = prFold
            else:
                prs = np.hstack((prs,prFold))

        aucVal = AllRes[1][-1]
        val = AllRes[2][-1]
        y_prs = np.mean(prs, axis=1)
        pltRoc.plot(x_map,y_prs,
                 #label = 'ROC-AUC: {}, calc-auc: {}'.format(aucVal, auc(x_map, y_prs)),
                 label='ROC-AUC: {} Rnd: {}'.format(aucVal,rndNum),
                 linewidth = 1.8 ,
                 fillstyle='none',color=cVals[rndCnt],dashes=lineSty[2])

    leg = pltRoc.legend(fancybox=True)
    axesRoc = pltRoc.gca()
    axesRoc.set_xlim([-0.02, 1.0])
    axesRoc.set_ylim([0.0, 1.02])
    pltRoc.ylabel('False Positive Rate (ROC curve)')
    pltRoc.xlabel('True Positive Rate')
    title = 'Receiver Operating Characteristic - {}'.format(lvl)
    pltRoc.title(title)
    pltRoc.legend(loc="lower right")
    # pltRoc.savefig('../../../ThesisWriteUp/fig'+'/'+clftype+'_FindThreshold_RocCurve_'+lvl+'.png')
    pltRoc.savefig(typeOrig+'_CombPlot_' + clftype + '_RocCurves_' + lvl + '.png')
    pltRoc.clf()
    pltRoc.close()

    pltPR = plt
    pltPR.style.use('ggplot')
    pltPR.figure()
    rndCnt = -1
    for rndNum in rndNums:
        rndCnt += 1
        AllRes = AllResRnds[rndNum]
        rndTypeFoldMat = rndTypeFoldMatRnds[rndNum]

        prs = []
        x_map = np.arange(0.0, 1.001, 0.001)
        x_map = x_map.reshape(len(x_map), 1)
        for fold in sorted(rndTypeFoldMat[type]):
        #for fold in [1,2]:
            lvl = re.split("[_]", type)[3]
            resultMat = []
            colNum = 0
            finds = ['rec', 'prec']
            for fnd in finds:
                outFind = dict()
                resultMat.append([])
                print('{} {}'.format(colNum,fnd))
                resultMat[colNum].append(lvl+'-'+fnd)
                axisValFlds = []
                axisVal = []
                for rec in rndTypeFoldMat[type][fold]:
                    if('rnd' in rec):
                        if(rec[1] == rndNum):
                            if(lvl in rec and finds[0] in rec and finds[1] in rec ):#and finds[2] in rec):
                                ind =[i for i in range(len(rec)) if rec[i] == fnd]
                                axisVal.append(rec[ind[0]+1])
                                #resultMat[colNum].append('{:}'.format(rec[ind[0] + 1]))
                axisVal = np.array(axisVal).reshape(len(axisVal),1)
                print('axisVal: {} fld: {}'.format(len(axisVal),fold))
                resultMat[colNum] =axisVal
                colNum += 1

            prFold = mapToAxisPR(x_map, resultMat)
            if len(prs) == 0:
                prs = prFold
            else:
                prs = np.hstack((prs,prFold))
            # plt.plot(resultMat[0],resultMat[1],
            #          linewidth = 1.8 ,
            #          fillstyle='none',color=cVals[0],dashes=lineSty[2])
            # plt.plot(x_map,prFold,
            #          linewidth = 1.8 ,
            #          fillstyle='none',color=cVals[1],dashes=lineSty[2])
        aucVal = AllRes[0][-1]
        val = AllRes[3][-1]
        y_prs = np.mean(prs, axis=1)
        pltPR.plot(x_map,y_prs,
                 #label = 'PR-AUC: {}, calc-auc: {}'.format(aucVal, auc(x_map, y_prs)),
                 label='PR-AUC: {} Rnd: {}'.format(aucVal,rndNum),
                 linewidth = 1.8 ,
                 fillstyle='none',color=cVals[rndCnt],dashes=lineSty[2])

    leg= pltPR.legend(fancybox=True)
    axesPR = pltPR.gca()

    axesPR.set_xlim([-0.02,1.0])
    axesPR.set_ylim([0.0,1.02])

    pltPR.ylabel('Precision (PR Curve)')
    pltPR.xlabel('Recall')
    title = 'Precision Recall - {}'.format(lvl)
    pltPR.title(title)
    pltPR.legend(loc="lower right")
    #pltPR.savefig('../../../ThesisWriteUp/fig'+'/'+clftype+'_FindThreshold_PrCurve_'+lvl+'.png')
    pltPR.savefig(typeOrig+'_CombPlot_' + clftype + '_PrCurves_' + lvl + '.png')
    pltPR.clf()
    pltPR.close()


