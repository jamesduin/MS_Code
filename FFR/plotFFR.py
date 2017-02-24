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




import numpy as np
from matplotlib.colors import LinearSegmentedColormap as lsc


def cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: map(lambda x: x[0], cdict[key]) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_list = np.unique(sum(step_dict.values(), []))
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(map(function, y0))
    y1 = np.array(map(function, y1))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    return lsc(name, step_dict, N=N, gamma=gamma)







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

resultsDir = 'runFFRR_Cst1/results'
#resultsDir = 'runFFRR_Cst16/results'
#resultsDir = '_results/results'

rndTypeSet = getRndTypeSet(resultsDir)
#rndTypeSet = {'FFR_0p5', 'FFR_0p6', 'FFR_0p2', 'FFR_0p1', 'FFR_1p0', 'FFR_0p3', 'FFR_0p0', 'FFR_0p4', 'FFR_0p7', 'FFR_0p9', 'FFR_0p8'}
#rndTypeSet = {'FFR_1p0'}#'FFR_0p0','FFR_0p1','FFR_0p2','FFR_0p3','FFR_0p4','FFR_0p5','FFR_0p6','FFR_0p7','FFR_0p8','FFR_0p9','FFR_1p10'}
rndTypeFoldMat = dict()
for type in rndTypeSet:
    foldMatrix = dict()
    for fold in range(1,11):
        for fname in glob.glob(resultsDir+'/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0]+'_'+rndType[1]
            typeNm = rndType[0]+'_'+type.replace('.','p')
            if (typeNm == instType and str(fold) == rndType[2] ):
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

cmap = plt.get_cmap('rainbow')

cNorm  = colors.Normalize(vmin=-0.0, vmax=1.1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
lineSty = [[2,1],[4,1],[8,1],[2,1],[4,1],
           [2, 1], [4, 1], [8, 1], [2, 1], [4, 1],[8, 1]]
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
        if prs == []:
            prs =prFold
        else:
            minRL = min(len(prs),len(prFold))
            prs = np.hstack((prs[:minRL],prFold[:minRL]))
            # if(len(prs) == len(prFold)):
            #     prs = np.hstack((prs, prFold))

    x_pr = np.array(range(1,len(prs)+1))
    y_pr = np.mean(prs, axis=1)
    colorVal = scalarMap.to_rgba(1-float(type))
    cVal = np.multiply(colorVal,(.9,.9,.9,1.0))
    plt.plot(x_pr,y_pr, label = 'FFR['+type+']',linewidth = 1.8 ,  color=cVal,dashes=lineSty[linInd] )

plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('FFR Method Fine Cost 1')
plt.legend(loc="lower right")
plt.savefig(resultsDir+'/FFR_PR.png')

