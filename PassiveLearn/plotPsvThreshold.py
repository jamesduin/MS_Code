import numpy as np
import matplotlib
matplotlib.use('Agg')
import pprint as pp
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
#os.chdir('FindThresholdSVM')  # FindThresholdSVM, FindThresholdLogReg
#os.chdir('FINAL/LogReg_All')
#os.chdir('FINAL/LogReg_All')
#os.chdir('FINAL/SVM_All')
#os.chdir('RocPR/LogRegSampleWeight')
#os.chdir('RocPR/LogRegNoSW')
os.chdir('RocPR/InvestigateRocPR')

clftype = 'LogReg' #LogReg, SVM

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

finds = ['pr','roc','acc','f1']
AllRes = []
colNum = 0
for fnd in finds:
    for type in rndTypeSet:
        outFind = dict()
        for lvl in ['coarse','fine']:
            AllRes.append([])
            print(colNum)
            AllRes[colNum].append(lvl+'-'+fnd)
            prFold = []
            for fold in sorted(rndTypeFoldMat[type]):
                for rec in rndTypeFoldMat[type][fold]:
                    #if(not isinstance(rec,str)):
                    if(fnd in rec and lvl in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        prFold.append(rec[ind[0]+1])
                        AllRes[colNum].append('{:.3f}'.format(rec[ind[0]+1]))
            prFold = np.array(prFold)
            #print(prFold)
            AllRes[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
            colNum += 1

print(AllRes)

lineSty = [[8,1],[4,1],[2,1],
           [8, 1], [4, 1], [2, 1],
           [8, 1], [4, 1], [2, 1],[4, 1],
           [8, 1]]
markSty = ['s','8','>',
           's','8','>',
           's','8','>','8',
           's']
markEvSty = [(1,8),(2,8),(3,8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8)]

cVals = [(0.90000000000000002, 0.25162433333706963, 0.12708553664078234, 1.0),
# (0.90000000000000002, 0.48317993787976221, 0.25162433333706957, 1.0),
# (0.86294117647058821, 0.67619870060118625, 0.3711206857265133, 1.0),
# (0.69352941176470584, 0.81992038432147962, 0.48784802082520451, 1.0),
# (0.53117647058823525, 0.89098219195263628, 0.58975546501210829, 1.0),
# (0.36882352941176472, 0.89098219195263628, 0.67984446124709452, 1.0),
# (0.2064705882352941, 0.8199203843214794, 0.75630966447410342, 1.0),
(0.037058823529411734, 0.67619870060118592, 0.81992038432147951, 1.0),
# (0.12529411764705883, 0.48317993787976199, 0.86410948083716543, 1.0),
# (0.28764705882352942, 0.25162433333706941, 0.89098219195263628, 1.0),
(0.45000000000000001, 0.0, 0.90000000000000002, 1.0)
 ]








#
finds = ['flsPos','truPos','ac','thresh']
for lvl in ['fine','coarse']:
    plt.figure()
    plt.style.use('ggplot')
    x_vals = []
    for fold in sorted(rndTypeFoldMat['Psv']):
    #for fold in [1]:
        resultMat = []
        colNum = 0
        for fnd in finds:
            for type in rndTypeSet:
                outFind = dict()
                #for lvl in ['coarse','fine']:

                resultMat.append([])
                print('{} {}'.format(colNum,fnd))
                axisValFlds = []
                #for fold in sorted(rndTypeFoldMat[type]):
                #for fold in [2]:
                axisVal = []
                for rec in rndTypeFoldMat[type][fold]:
                    if(lvl in rec and finds[0] in rec and finds[1] in rec and finds[2] in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        axisVal.append(rec[ind[0]+1])
                        #resultMat[colNum].append('{:}'.format(rec[ind[0] + 1]))
                axisVal = np.array(axisVal).reshape(len(axisVal),1)
                print('axisVal: {} fld: {}'.format(len(axisVal),fold))
                if axisValFlds == []:
                    axisValFlds = axisVal
                else:
                    axisValFlds = np.hstack((axisValFlds,axisVal))
                #resultMat[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
                #resultMat[colNum] = np.mean(axisValFlds,axis=1)
                resultMat[colNum] =axisValFlds
                colNum += 1
        # print(len(resultMat[0]))
        # print(len(resultMat[1]))
        # print(len(resultMat[2]))
        # print(resultMat[0])
        # print(resultMat[1])
        #pp.pprint(resultMat)
        #
        # print(resultMat[0][1:20])
        # print(resultMat[2][1:20])
        resComb = np.hstack((resultMat[0],resultMat[1]))
        resComb = np.hstack((resComb, resultMat[2]))
        resComb = np.hstack((resComb, resultMat[3]))

        prev = 0
        cur = 0
        for i,res in enumerate(resComb):
            cur = i
            if(float(res[3])<0.0):
                break
            prev = i
        #print('prev {}, cur{}'.format(prev,cur))
        #print(resComb[prev])
        #print(resComb[cur])
        y = np.array([float(resComb[prev][0]), float(resComb[cur][0])])
        x = np.array([float(resComb[prev][3]), float(resComb[cur][3])])
        m = (y[0] - y[1]) / (x[0] - x[1])
        b = y[0] - m * x[0]
        x_val = m * 0.0 + b
        #print('x_val {}'.format(x_val))
        x_vals.append(x_val)

        if fold == 1:
            if(lvl == 'coarse'):
                auc = AllRes[2][-1]
            elif(lvl == 'fine'):
                auc = AllRes[3][-1]

            if(lvl == 'coarse'):
                val = AllRes[4][-1]
            elif(lvl == 'fine'):
                val = AllRes[5][-1]

            plt.plot(resultMat[0][:],resultMat[1][:],
                     label = 'ROC-AUC: {}'.format(auc),
                     linewidth = 1.8 ,
                     fillstyle='none',color=cVals[0],dashes=lineSty[2])
            plt.plot(resultMat[0][:],resultMat[2][:],
                     label = 'Accuracy: {}'.format(val),
                     linewidth = 1.8 ,
                     fillstyle='none',color=cVals[1],dashes=lineSty[2])
        else:
            plt.plot(resultMat[0][:],resultMat[1][:],linewidth = 1.8 ,
                     fillstyle='none',color=cVals[0],dashes=lineSty[2])
            plt.plot(resultMat[0][:],resultMat[2][:],linewidth = 1.8 ,
                     fillstyle='none',color=cVals[1],dashes=lineSty[2])

    plt.plot(np.mean(x_vals),float(val.replace('avg ','')),label='Chosen Threshold',
    fillstyle='none', color=cVals[2], marker='x', markersize=10,
             markeredgecolor=cVals[2],markeredgewidth=3.0)

    leg= plt.legend(fancybox=True)
    axes = plt.gca()

    axes.set_xlim([-0.02,1.0])
    axes.set_ylim([0.0,1.02])
    #axes.set_ylim([0.838, 0.87])

    plt.ylabel('False Positive Rate (ROC curve) / Accuracy')
    plt.xlabel('True Positive Rate')
    title = 'Receiver Operating Characteristic - {}'.format(lvl)
    plt.title(title)
    plt.legend(loc="lower right")
    #plt.savefig('../../../ThesisWriteUp/fig'+'/'+clftype+'_FindThreshold_RocCurve_'+lvl+'.png')
    plt.savefig(clftype + '_RocCurves_' + lvl + '.png')




finds = ['rec','prec','fmes','thresh']
for lvl in ['coarse','fine']:
    plt.figure()
    plt.style.use('ggplot')
    x_vals =[]
    for fold in sorted(rndTypeFoldMat['Psv']):
    #for fold in [1]:
        resultMat = []
        colNum = 0
        for fnd in finds:
            for type in rndTypeSet:
                outFind = dict()
                #for lvl in ['coarse','fine']:

                resultMat.append([])
                print('{} {}'.format(colNum,fnd))
                resultMat[colNum].append(lvl+'-'+fnd)
                axisValFlds = []
                #for fold in sorted(rndTypeFoldMat[type]):
                #for fold in [2]:
                axisVal = []
                for rec in rndTypeFoldMat[type][fold]:
                    if(lvl in rec and finds[0] in rec and finds[1] in rec and finds[2] in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        axisVal.append(rec[ind[0]+1])
                        #resultMat[colNum].append('{:}'.format(rec[ind[0] + 1]))
                axisVal = np.array(axisVal).reshape(len(axisVal),1)
                print('axisVal: {} fld: {}'.format(len(axisVal),fold))
                if axisValFlds == []:
                    axisValFlds = axisVal
                else:
                    axisValFlds = np.hstack((axisValFlds,axisVal))
                #resultMat[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
                #resultMat[colNum] = np.mean(axisValFlds,axis=1)
                resultMat[colNum] =axisValFlds
                colNum += 1
        #print(len(resultMat[0]))
        #print(len(resultMat[1]))
        #print(len(resultMat[2]))

        resComb = np.hstack((resultMat[0], resultMat[1]))
        resComb = np.hstack((resComb, resultMat[2]))
        resComb = np.hstack((resComb, resultMat[3]))
        #pp.pprint(resComb[:5])
        prev = 0
        cur = 0
        for i, res in enumerate(resComb):
            cur = i
            if (float(res[3]) > 0.0):
                break
            prev = i
        #print('prev {}, cur {}'.format(prev, cur))
        #print(resComb[prev])
        #print(resComb[cur])
        y = np.array([float(resComb[prev][0]), float(resComb[cur][0])])
        x = np.array([float(resComb[prev][3]), float(resComb[cur][3])])
        m = (y[0] - y[1]) / (x[0] - x[1])
        b = y[0] - m * x[0]
        x_val = m * 0.0 + b
        #print('x_val {}'.format(x_val))
        x_vals.append(x_val)

        if fold == 1:
            if(lvl == 'coarse'):
                auc = AllRes[0][-1]
            elif(lvl == 'fine'):
                auc = AllRes[1][-1]
            if (lvl == 'coarse'):
                val = AllRes[6][-1]
            elif (lvl == 'fine'):
                val = AllRes[7][-1]

            plt.plot(resultMat[0][:],resultMat[1][:],
                     label='PR-AUC: {}'.format(auc),
                     linewidth = 1.8 ,
                     fillstyle='none',color=cVals[0],dashes=lineSty[2])
            plt.plot(resultMat[0][:],resultMat[2][:],
                     label = 'F-measure: {}'.format(val),
                     linewidth = 1.8 ,
                     fillstyle='none',color=cVals[1],dashes=lineSty[2])
        else:
            plt.plot(resultMat[0][:],resultMat[1][:],linewidth = 1.8 ,
                     fillstyle='none',color=cVals[0],dashes=lineSty[2])
            plt.plot(resultMat[0][:],resultMat[2][:],linewidth = 1.8 ,
                     fillstyle='none',color=cVals[1],dashes=lineSty[2])

    plt.plot(np.mean(x_vals),float(val.replace('avg ','')),label='Chosen Threshold',
    fillstyle='none', color=cVals[2], marker='x', markersize=10,
             markeredgecolor=cVals[2],markeredgewidth=3.0)

    leg= plt.legend(fancybox=True)
    axes = plt.gca()

    axes.set_xlim([-0.02,1.0])
    axes.set_ylim([0.0,1.02])
    #axes.set_ylim([0.838, 0.87])

    plt.ylabel('Precision (PR Curve) / F-measure')
    plt.xlabel('Recall')
    title = 'Precision Recall - {}'.format(lvl)
    plt.title(title)
    plt.legend(loc="lower right")
    #plt.savefig('../../../ThesisWriteUp/fig'+'/'+clftype+'_FindThreshold_PrCurve_'+lvl+'.png')
    plt.savefig(clftype + '_PrCurves_' + lvl + '.png')