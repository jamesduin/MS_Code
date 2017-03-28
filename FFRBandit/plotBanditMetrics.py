import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
import matplotlib.colors as colors
import matplotlib.cm as cmx
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
        #print(rndType[0] + '_' + rndType[1])
        #rndTypeSet.add(rndType[0] + '_' + rndType[1])
        rndTypeSet.add(rndType[1].replace('p','.'))
    print(rndTypeSet)
    return rndTypeSet

#resultsDir = 'runFFRParam_Cst16/results'
#resultsDir = 'runFFRParam_Cst8/results'
#resultsDir = 'BanditTest/results'
#resultsDir = 'Bandit_RandEqual/results'
#resultsDir = 'Bandit_TrueEqualStay_Cst8/results'
#resultsDir = 'Bandit_RandEqual_Cst1/results'
#resultsDir = 'BANDIT_1p1/results'
#resultsDir = 'BANDIT_2p0/results'
#resultsDir = 'BANDIT_4p0/results'
resultsDir = 'Bandit_RandEqual_Cst8/results'
#resultsDir = 'BANDIT_16p0/results'

#Xlims = [0, 500]
#Xlims = [20,60]
Xlims = [0,180]
#Ylims = [0.827, 0.863]
cost = 8
#resultsDir = '_results/results'





rndTypeSet = getRndTypeSet(resultsDir)
#rndTypeSet = {'FFR_0p5', 'FFR_0p6', 'FFR_0p2', 'FFR_0p1', 'FFR_1p0', 'FFR_0p3', 'FFR_0p0', 'FFR_0p4', 'FFR_0p7', 'FFR_0p9', 'FFR_0p8'}
#rndTypeSet = {'FFR_1p0'}#'FFR_0p0','FFR_0p1','FFR_0p2','FFR_0p3','FFR_0p4','FFR_0p5','FFR_0p6','FFR_0p7','FFR_0p8','FFR_0p9','FFR_1p10'}
#rndTypeSet = {'1.0'}#{'0.6', '0.8', '0.2', '0.4', '0.0',  }
rndTypeFoldMat = dict()
for type in rndTypeSet:
    foldMatrix = dict()
    for fold in range(1,11):
    #for fold in [3,4,5,6,7,8,9]:
        for fname in glob.glob(resultsDir+'/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0] + '_' + rndType[1]
            typeNm = rndType[0] + '_' + type.replace('.', 'p')
            if (typeNm == instType and str(fold) == rndType[2]):
                # print(fold)
                # print(instType)
                foldMatrix[fold] = []
                results = []
                try:
                    results = pickle.load(open(fname, 'rb'))
                except EOFError:
                    pass
                for result in results:
                    foldMatrix[fold].append(result)
            # file = re.split("[/\.]", fname)[-2]
            # rndType = re.split("[_]", file)
            # instType = rndType[0]+'_'+rndType[1]
            # typeNm = rndType[0]+'_'+type
            # print('instType {}'.format(instType))
            # print('typeNm {}'.format(typeNm))
            # if (typeNm == instType and str(fold) == rndType[1] ):
            #     # print(fold)
            #     # print(instType)
            #     foldMatrix[fold] = []
            #     results = []
            #     try:
            #         results = pickle.load(open(fname, 'rb'))
            #     except EOFError:
            #         pass
            #     for result in results:
            #         foldMatrix[fold].append(result)
    rndTypeFoldMat[type] = foldMatrix

max = []
for fold in foldMatrix:
    max.append(len(foldMatrix[fold]))
#print(np.max(max))

for type in rndTypeFoldMat:
    ### print out the folds
    typeNm = rndType[0] + '_' + type.replace('.', 'p')
    f = open(resultsDir+'/_' + typeNm + '.txt', 'w')
    for fold in rndTypeFoldMat[type]:
        for rec in rndTypeFoldMat[type][fold]:
            f.write(str(rec)+'\n')
    f.close()

plt.figure()
#with plt.style.context('fivethirtyeight'):
plt.style.use('ggplot')

cmap = plt.get_cmap('rainbow')

cNorm  = colors.Normalize(vmin=-0.0, vmax=1.1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
lineSty = [[8,1],[4,1],[2,1],
           [8, 1], [4, 1], [2, 1],
           [8, 1], [4, 1], [2, 1],[4, 1],
           [8, 1],[8, 1]]
markSty = ['s','8','>',
           's','8','>',
           's','8','>','8',
           's','s']
markEvSty = [(1,8),(2,8),(3,8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (1, 8)]

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
(0.45000000000000001, 0.0, 0.90000000000000002, 1.0),
 (0.0, 0.0, 0.0, 1.0)]
# cVals = [
# (255.*.9,255*.25, 255*.12, 1.0),
# (254., 50., 7., 1.0),
# (254., 102., 0., 1.0),
# (254., 169., 0., 1.0),
# (255., 209., 32., 1.0),
# (10., 207., 0., 1.0),
# (7., 142., 23., 1.0),
# (11., 156., 152., 1.0),
# (10., 0., 214., 1.0),
# (67., 0., 104., 1.0),
# (0., 0., 0., 1.0)
# ]



for linInd,type in enumerate(sorted(rndTypeSet)):
    prs = []
    for fold in sorted(rndTypeFoldMat[type]):
        prFold = []
        for rec in rndTypeFoldMat[type][fold]:
            #if(not isinstance(rec,str)):co
            if('comb_pr'in rec):
                ind =[i for i in range(len(rec)) if rec[i] == 'comb_pr']
                prFold.append(rec[ind[0]+1])
        prFold = np.array(prFold).reshape(len(prFold),1)
        #print(prFold[:1])
        if len(prs)==0:
            prs =prFold
        else:
           # if(len(prFold)!=0):
            minRL = min(len(prs),len(prFold))
            prs = np.hstack((prs[:minRL],prFold[:minRL]))
            # if(len(prs) == len(prFold)):
                #     prs = np.hstack((prs, prFold))

    x_pr = np.array(range(1,len(prs)+1))
    y_pr = np.mean(prs, axis=1)
    #cVal = tuple(np.multiply(cVals[linInd],(1/270,1/270,1/270,1.0)))
    cVal = cVals[linInd]
    print('FFR: {},{},{}'.format(type,x_pr[-1],y_pr[-1]))
    plt.plot(x_pr,y_pr, label = 'FFR['+type+']',linewidth = 1.8 ,
             fillstyle='none',
             color=cVal,dashes=lineSty[linInd],
             # marker=markSty[linInd],
             # markersize=6,
             # markeredgecolor=cVal,
             # markeredgewidth=1.0,
             # markevery=markEvSty[linInd]
             )
    leg= plt.legend(fancybox=True)
    axes = plt.gca()

    axes.set_xlim(Xlims)
    #axes.set_ylim([0.842, 0.875])
    #axes.set_ylim(Ylims)

plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
title = 'FFR Method Fine Cost '+str(cost)
# if(Xlims[1] == 500):
#     title = title+' - '+str(Xlims[1])+' Rounds'
# if(Xlims[1] == 60):
#     title = title + ' - '+ str(Xlims[0])+' to '+ str(Xlims[1]) + ' Rounds'
plt.title(title)
plt.legend(loc="lower right")
#plt.savefig('../ThesisWriteUp/fig'+'/ParamsFFR_PR_Cost'+str(cost)+'_rnds'+str(Xlims[0])+'_'+str(Xlims[1])+'.png')
plt.savefig(resultsDir+'/ParamsFFR_PR_Cost'+str(cost)+'_rnds'+str(Xlims[0])+'_'+str(Xlims[1])+'.png')