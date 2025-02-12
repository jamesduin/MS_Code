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
os.chdir('runActPassLogReg')

clftype = 'LogReg' #LogReg, SVM
#rndNums = [20,30,40,50,60]
rndNums = [20]


#type = 'active_fine'
rndTypes = ['active_fine']#,'active_coarse']
#rndTypes = ['active_coarse']
logLoc = 'thresh_100_'

for rndNum in rndNums:
    for type in rndTypes:
        resultsDir = 'results'
        rndTypeFoldMat = dict()
        foldMatrix = dict()
        for fold in range(1,11):
            for fname in glob.glob(resultsDir+'/*.res'):
                file = re.split("[/\.]", fname)[-2]
                rndType = re.split("[_]", file)
                instType = rndType[0]+'_'+rndType[1]
                if (type == instType and str(fold) == rndType[2] ):
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
            AllRes[colNum].append(lvl+'-'+fnd)
            prFold = []
            for fold in sorted(rndTypeFoldMat[type]):
                for rec in rndTypeFoldMat[type][fold]:
                    #if(not isinstance(rec,str)):
                    if('rnd' in rec):
                        if(rec[1] == rndNum):
                            if(fnd in rec and lvl in rec):
                                ind =[i for i in range(len(rec)) if rec[i] == fnd]
                                prFold.append(rec[ind[0]+1])
                                AllRes[colNum].append('{:.3f}'.format(rec[ind[0]+1]))
            prFold = np.array(prFold)
            #print(prFold)
            AllRes[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
            colNum += 1

        print(AllRes)



        resultsDir = 'thresh'
        type = logLoc+type

        rndTypeFoldMat = dict()
        foldMatrix = dict()
        for fold in range(1,11):
            for fname in glob.glob(resultsDir+'/*.res'):
                file = re.split("[/\.]", fname)[-2]
                rndType = re.split("[_]", file)
                instType = rndType[0]+'_'+rndType[1]+'_'+rndType[2]+'_'+rndType[3]
                if (type == instType and str(fold) == rndType[4] ):
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
















        for type in rndTypeFoldMat:
            plt.figure()
            plt.style.use('ggplot')
            x_axis = []
            prs = []
            #for fold in sorted(rndTypeFoldMat[type]):
            for fold in [1,2]:
                lvl = re.split("[_]", type)[3]
                resultMat = []
                colNum = 0
                finds = ['flsPos', 'truPos']
                for fnd in finds:
                    #for type in rndTypeSet:
                    outFind = dict()
                    #for lvl in ['coarse','fine']:

                    resultMat.append([])
                    print('{} {}'.format(colNum,fnd))
                    axisValFlds = []
                    #for fold in sorted(rndTypeFoldMat[type]):
                    #for fold in [2]:
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

                if(fold == 1):
                    x_axis = resultMat[0]
                    prs = resultMat[1]
                else:
                    prFold = []
                    for i, res in enumerate(x_axis):
                        prev = 0
                        cur = 0
                        for j, old in enumerate(resultMat[0]):
                            cur = j
                            if(res<=old):
                                break
                            prev = j
                        y = np.array([resultMat[1][prev], resultMat[1][cur]])
                        x = np.array([resultMat[0][prev],resultMat[0][cur]])
                        m = (y[0] - y[1]) / (x[0] - x[1])
                        b = y[0] - m * x[0]
                        pr_inter = m * res + b
                        print('x_axis {}'.format(res))
                        print('pr_inter {}'.format(pr_inter))
                        print('point1 {}, {}'.format(resultMat[0][prev], resultMat[1][prev]))
                        print('point2 {}, {}'.format(resultMat[0][cur], resultMat[1][cur]))

                        prFold.append(pr_inter)
                    prFold = np.array(prFold).reshape(len(prFold), 1)
                    plt.plot(x_axis, prFold,
                             linewidth=1.8,
                             fillstyle='none', color=cVals[0], dashes=lineSty[2])
                    plt.plot(resultMat[0], resultMat[1],
                             linewidth=1.8,
                             fillstyle='none', color=cVals[1], dashes=lineSty[2])


            auc = AllRes[1][-1]
            val = AllRes[2][-1]

            plt.plot(x_axis,prs,
                     label = 'ROC-AUC: {}'.format(auc),
                     linewidth = 1.8 ,
                     fillstyle='none',color=cVals[0],dashes=lineSty[2])

                #
                # if len(prs) == 0:
                #     prs = prFold
                # else:
                #     minRL = min(len(prs), len(prFold))
                #     prs = np.hstack((prs[:minRL], prFold[:minRL]))


                # if fold == 1:
                #     auc = AllRes[1][-1]
                #     val = AllRes[2][-1]
                #
                #     plt.plot(resultMat[0][:],resultMat[1][:],
                #              label = 'ROC-AUC: {}'.format(auc),
                #              linewidth = 1.8 ,
                #              fillstyle='none',color=cVals[0],dashes=lineSty[2])
                # else:
                #     plt.plot(resultMat[0][:],resultMat[1][:],linewidth = 1.8 ,
                #              fillstyle='none',color=cVals[0],dashes=lineSty[2])

            leg= plt.legend(fancybox=True)
            axes = plt.gca()

            axes.set_xlim([-0.02,1.0])
            axes.set_ylim([0.0,1.02])
            #axes.set_ylim([0.838, 0.87])

            #plt.ylabel('False Positive Rate (ROC curve) / Accuracy')
            plt.ylabel('False Positive Rate (ROC curve)')
            plt.xlabel('True Positive Rate')
            title = 'Receiver Operating Characteristic - {}'.format(lvl)
            plt.title(title)
            plt.legend(loc="lower right")
            #plt.savefig('../../../ThesisWriteUp/fig'+'/'+clftype+'_FindThreshold_RocCurve_'+lvl+'.png')
            #plt.savefig('RndNum_'+str(rndNum)+'_'+clftype + '_RocCurves_' + lvl + '.png')
            plt.savefig('Inter_AVG_RndNum_' + str(rndNum) + '_' + clftype + '_RocCurves_' + lvl + '.png')




        #
        #
        # for type in rndTypeFoldMat:
        #     plt.figure()
        #     plt.style.use('ggplot')
        #     x_vals =[]
        #     for fold in sorted(rndTypeFoldMat[type]):
        #     #for fold in [1]:
        #         lvl = re.split("[_]", type)[3]
        #         resultMat = []
        #         colNum = 0
        #         finds = ['rec', 'prec']
        #         for fnd in finds:
        #             #for type in rndTypeSet:
        #             outFind = dict()
        #             #for lvl in ['coarse','fine']:
        #
        #             resultMat.append([])
        #             print('{} {}'.format(colNum,fnd))
        #             resultMat[colNum].append(lvl+'-'+fnd)
        #             axisValFlds = []
        #             #for fold in sorted(rndTypeFoldMat[type]):
        #             #for fold in [2]:
        #             axisVal = []
        #             for rec in rndTypeFoldMat[type][fold]:
        #                 if('rnd' in rec):
        #                     if(rec[1] == rndNum):
        #                         if(lvl in rec and finds[0] in rec and finds[1] in rec ):#and finds[2] in rec):
        #                             ind =[i for i in range(len(rec)) if rec[i] == fnd]
        #                             axisVal.append(rec[ind[0]+1])
        #                             #resultMat[colNum].append('{:}'.format(rec[ind[0] + 1]))
        #             axisVal = np.array(axisVal).reshape(len(axisVal),1)
        #             print('axisVal: {} fld: {}'.format(len(axisVal),fold))
        #             resultMat[colNum] =axisVal
        #             colNum += 1
        #
        #         if fold == 1:
        #             auc = AllRes[0][-1]
        #             val = AllRes[3][-1]
        #
        #             plt.plot(resultMat[0][:],resultMat[1][:],
        #                      label='PR-AUC: {}'.format(auc),
        #                      linewidth = 1.8 ,
        #                      fillstyle='none',color=cVals[0],dashes=lineSty[2])
        #         else:
        #             plt.plot(resultMat[0][:],resultMat[1][:],linewidth = 1.8 ,
        #                      fillstyle='none',color=cVals[0],dashes=lineSty[2])
        #     leg= plt.legend(fancybox=True)
        #     axes = plt.gca()
        #
        #     axes.set_xlim([-0.02,1.0])
        #     axes.set_ylim([0.0,1.02])
        #
        #     plt.ylabel('Precision (PR Curve)')
        #     plt.xlabel('Recall')
        #     title = 'Precision Recall - {}'.format(lvl)
        #     plt.title(title)
        #     plt.legend(loc="lower right")
        #     #plt.savefig('../../../ThesisWriteUp/fig'+'/'+clftype+'_FindThreshold_PrCurve_'+lvl+'.png')
        #     #plt.savefig('RndNum_'+str(rndNum)+'_'+clftype + '_PrCurves_' + lvl + '.png')
        #     plt.savefig('Inter_AVG_RndNum_' + str(rndNum) + '_' + clftype + '_PrCurves_' + lvl + '.png')