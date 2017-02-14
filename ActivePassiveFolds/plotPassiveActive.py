import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'Users'):
    dataDir = '../'
else:
    os.chdir('/work/scott/jamesd/')
    dataDir = '/home/scott/jamesd/MS_Code/'

plt.figure()
#with plt.style.context('fivethirtyeight'):
plt.style.use('ggplot')


#resultsDir = 'resultsSclBy1_15/results'
#resultsDir = 'resultsRBFsclBy1_15/results'
resultsDir = 'resultsPart11SclBy1_15/results'
#resultsDir = 'results'

rndTypeSet = set()
for fname in glob.glob(resultsDir+'/*.res'):
    #print(fname)
    file = re.split("[/\.]", fname)[-2]
    #print(file)
    rndType = re.split("[_]", file)
    #print(rndType)
    print(rndType[0]+'_'+rndType[1])
    rndTypeSet.add(rndType[0]+'_'+rndType[1])
print(rndTypeSet)

for type in rndTypeSet:
    prs = []
    foldMatrix = dict()
    for fold in range(1,11):
        #fold = 11-fold
        for fname in glob.glob(resultsDir+'/*.res'):
            file = re.split("[/\.]", fname)[-2]
            rndType = re.split("[_]", file)
            instType = rndType[0]+'_'+rndType[1]
            #if(type == instType and str(fold) == rndType[2] and fold <=7):
            if (type == instType and str(fold) == rndType[2] ):
                pr_row = []
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
                    fnd = False
                    for item in result:
                        if (fnd):
                            pr_row.append(item)
                            fnd = False
                        elif item == 'pr':
                            fnd = True
                if prs == []:
                    prs = np.array(pr_row).reshape(len(pr_row),1)
                else:
                    size = prs.shape[0]
                    row = pr_row[:size]
                    cntInd = len(row)
                    y_tmp = np.mean(prs, axis=1)
                    while(len(row)!= size):

                        row.append(y_tmp[cntInd])
                        cntInd+=1
                    pr = np.array(row).reshape(len(row),1)
                    prs = np.hstack((prs,pr))
        f = open(resultsDir+'/_' + type + '.txt', 'w')

        startFold = 0
        for fold in foldMatrix:
            startFold = fold
            break
        if(startFold != 0):
            for i in range(len(foldMatrix[startFold])):
                for fold in foldMatrix:
                    try:
                        f.write(str(foldMatrix[fold][i])+'\n')
                    except IndexError:
                        pass

    # print((prs[0][0]+prs[0][1]+prs[0][2])/3)
    x = np.array(range(1,len(prs)+1))
    y = np.mean(prs, axis=1)
    #print(x)
    #print(y)
    plt.plot(x,y, label = 'avg_'+type)

plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('Active vs. Passive Learning')
plt.legend(loc="lower right")
plt.savefig(resultsDir+'/ActiveVsPassive.png')
