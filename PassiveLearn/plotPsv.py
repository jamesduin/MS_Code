import numpy as np
import matplotlib
matplotlib.use('Agg')
import pprint as pp
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os


#dir = 'CostGamma/SVM_OrigWithFtune/results'
#dir = 'CostGamma/SVM_Cp1/results'
#dir = 'CostGamma/SVM_C1_Gp0029674/results'
#dir = 'CostGamma/SVM_C2_Gp0029674/results'
#dir = 'CostGamma/SVM_Cp1_Gp0029674/results'
#dir = 'CostGamma/SVM_Cp05_Gp0029674/results'
#dir = 'CostGamma/SVM_Cp15_Gp0029674/results'
#dir = 'CostGamma/SVM_Cp2_Gp0029674/results'
#dir = 'CostGamma/SVM_C1_Gp002/results'
#dir = 'CostGamma/SVM_C1_Gp001/results'
#dir = 'CostGamma/SVM_C1_Gp0005/results'
#dir = 'CostGamma/SVM_C1_Gp0015/results'
# dir = 'CostGamma/SVM_C1_Gp0025/results'
#dir = 'CostGamma/SVM_C1_Gp0035/results'
#dir = 'CostGamma/SVM_Cp1_Gp003/results'
#dir = 'CostGamma/SVM_Cp1_Gp0025/results'
#dir = 'CostGamma/SVM_Cp1_Gp002/results'
#dir = 'CostGamma/SVM_Cp1_Gp001/results'
#dir = 'CostGamma/SVM_Cp05_Gp002/results'
#dir = 'CostGamma/SVM_Cp15_Gp002/results'
#dir = 'CostGamma/SVM_Cp15_Gp001/results'
#dir = 'CostGamma/SVM_Cp5_Gp002/results'
#dir = 'CostGamma/SVM_Cp3_Gp002/results'
#dir = 'CostGamma/SVM_Cp15_Gp002_tolp00001/results'
#dir = 'CostGamma/SVM_C1_Gp0025_tolp00001/results'
#dir = 'ClassWeight/SVM_All/results'
#dir = 'ClassWeight/SVM_AllQuick/results'
#dir = 'FINAL/LogReg_00001/results'
#dir = 'Tolerance/LogReg_00001Redo/results'
#dir = 'RocPR/LogRegOrig/results'
#dir = 'RocPR/LogReg_NoSW/results'
#dir = 'RocPR/LogReg_NoSW_DropFalse/results'
#dir = 'RocPR/LogReg_DropFalse/results'
dir = 'Kernel/RbfOrig/results'
#dir = 'Kernel/Linear/results'
#dir = 'Kernel/polyDeg3/results'
#dir = 'Kernel/polyDeg6/results'
#dir = 'Kernel/sigmoid/results'

os.chdir(dir)
tableName = re.split('[/]',dir)[1].replace('_','-')

def getRndTypeSet():
    rndTypeSet = set()
    for fname in glob.glob('*.res'):
        file = re.split("[/\.]", fname)[-2]
        rndType = re.split("[_]", file)
        #print(rndType[0])
        rndTypeSet.add(rndType[0])
    #print(rndTypeSet)
    return rndTypeSet

rndTypeSet = getRndTypeSet()

rndTypeFoldMat = dict()
for type in rndTypeSet:
    foldMatrix = dict()
    for fold in range(1,11):
        for fname in glob.glob('*.res'):
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
    f = open('_' + type + '.txt', 'w')
    for fold in rndTypeFoldMat[type]:
        for rec in rndTypeFoldMat[type][fold]:
            f.write(str(rec)+'\n')
    f.close()



def printResultMat(f,resultMat):
    f.write('\FloatBarrier\n')
    f.write('\\begin{table}[h]\n')
    f.write('\centering\n')
    f.write('\\begin{tabular}{')
    for i in range(len(resultMat)):
        f.write('|l|')
    f.write('}\\toprule\n')
    for i, row in enumerate(resultMat[0]):
        for j, col in enumerate(resultMat[:-1]):
            if (isinstance(resultMat[j][i], str)):
                f.write('{} & '.format(resultMat[j][i]))
        if(i == 0):
            f.write('{} \\\\ \\midrule'.format(resultMat[-1][i]))
        elif(i == len(resultMat[0]) -1 ):
            f.write('{} \\\\ \\bottomrule'.format(resultMat[-1][i]))
        else:
            f.write('{} \\\\'.format(resultMat[-1][i]))
        f.write('\n')
    f.write('\end{tabular}\n')
    f.write('\caption{'+tableName+'}\n')
    f.write('\label{tab:'+tableName+'}\n')
    f.write('\end{table}\n')
    f.write('\FloatBarrier\n')
    f.write('\n')



finds = ['pr','roc','acc','f1']
resultMat = []
colNum = 0
for fnd in finds:
    for type in rndTypeSet:
        outFind = dict()
        for lvl in ['coarse','fine']:
            resultMat.append([])
            print(colNum)
            resultMat[colNum].append(lvl+'-'+fnd)
            prFold = []
            for fold in sorted(rndTypeFoldMat[type]):
                for rec in rndTypeFoldMat[type][fold]:
                    #if(not isinstance(rec,str)):
                    if(fnd in rec and lvl in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        prFold.append(rec[ind[0]+1])
                        resultMat[colNum].append('{:.3f}'.format(rec[ind[0]+1]))
            prFold = np.array(prFold)
            #print(prFold)
            resultMat[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
            colNum += 1
print(resultMat)
f = open('output.txt', 'w')
printResultMat(f,resultMat)




finds = ['tn','fp','fn','tp']
resultConf = []
colNum = 0
for fnd in finds:
    for type in rndTypeSet:
        outFind = dict()
        for lvl in ['coarse','fine']:
            resultConf.append([])
            print(colNum)
            resultConf[colNum].append(lvl+'-'+fnd)
            prFold = []
            for fold in sorted(rndTypeFoldMat[type]):
                for rec in rndTypeFoldMat[type][fold]:
                    #if(not isinstance(rec,str)):
                    if(fnd in rec and lvl in rec):
                        ind =[i for i in range(len(rec)) if rec[i] == fnd]
                        prFold.append(rec[ind[0]+1])
                        resultConf[colNum].append('{}'.format(rec[ind[0]+1]))
            prFold = np.array(prFold)
            #print(prFold)
            resultConf[colNum].append('avg {:.1f}'.format(np.mean(prFold)))
            colNum += 1
print(resultConf)
printResultMat(f,resultConf)



table = []
for lvl in ['coarse']:
    table.append(['title']+[lvl])
    for col,colItem in enumerate(resultMat):
        if(lvl in resultMat[col][0]):
            table.append([resultMat[col][0].replace(lvl+'-','')]+[resultMat[col][-1].replace('avg ','')])

tableCol = 0
for lvl in ['fine']:
    table[tableCol].append(lvl)
    tableCol += 1
    for col,colItem in enumerate(resultMat):
        if(lvl in resultMat[col][0]):
            table[tableCol].append(resultMat[col][-1].replace('avg ',''))
            tableCol += 1


print(table)




tmpCol = ['conf (tn/fn)','','','','']
for col,colItem in enumerate(resultConf):
    if('coarse-tn' in resultConf[col][0]):
        tmpCol[1] = resultConf[col][-1].replace('avg ','')
    if ('coarse-fn' in resultConf[col][0]):
        tmpCol[2] = resultConf[col][-1].replace('avg ', '')
    if('fine-tn' in resultConf[col][0]):
        tmpCol[3] = resultConf[col][-1].replace('avg ','')
    if ('fine-fn' in resultConf[col][0]):
        tmpCol[4] = resultConf[col][-1].replace('avg ', '')
table.append([tmpCol[0]]+['( {} / {} )'.format(tmpCol[1],tmpCol[2])]+
             ['( {} / {} )'.format(tmpCol[3], tmpCol[4])])

tmpCol = ['conf (fp/tp)','','','','']
for col,colItem in enumerate(resultConf):
    if('coarse-fp' in resultConf[col][0]):
        tmpCol[1] = resultConf[col][-1].replace('avg ','')
    if ('coarse-tp' in resultConf[col][0]):
        tmpCol[2] = resultConf[col][-1].replace('avg ', '')
    if('fine-fp' in resultConf[col][0]):
        tmpCol[3] = resultConf[col][-1].replace('avg ','')
    if ('fine-tp' in resultConf[col][0]):
        tmpCol[4] = resultConf[col][-1].replace('avg ', '')
table.append([tmpCol[0]]+['( {} / {} )'.format(tmpCol[1],tmpCol[2])]+
             ['( {} / {} )'.format(tmpCol[3], tmpCol[4])])

print(table)
printResultMat(f,table)
f.close()
