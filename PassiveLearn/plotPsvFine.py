import numpy as np
import matplotlib
matplotlib.use('Agg')
import pprint as pp
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os

#dir = 'Original/SVMDef/results'
#dir = 'ScalingDim/SVMStandard/results'
#dir = 'ScalingDim/SVMMinMax/results'
#dir = 'ScalingDim/SVMNorm/results'
#dir = 'ScalingDim/SVMSel25/results'
#dir = 'ScalingDim/SVMSel50/results'
#dir = 'ScalingDim/SVMSel75/results'
#dir = 'DecFcnShape/SVM_ovo/results'
#dir = 'DecFcnShape/SVM_ovr/results'
#dir = 'Original/LogRegDef/results'
#dir = 'ScalingDim/LogRegStandard/results'
#dir = 'ScalingDim/LogRegMinMax/results'
#dir = 'ScalingDim/LogRegNorm/results'
#dir = 'ScalingDim/LogRegSel25/results'
#dir = 'ScalingDim/LogRegSel50/results'
#dir = 'ScalingDim/LogRegSel75/results'
#dir = 'ClassWeight/LogRegWtOrig/results'
#dir = 'CostGamma/LogRegWtOrig_Cp1/results'
#dir = 'CostGamma/LogRegWtOrig_C10/results'
#dir = 'CostGamma/LogRegWt10_C1/results'
#dir = 'CostGamma/LogRegWt10_Cp1/results'
#dir = 'CostGamma/LogRegWt10_C10/results'
#dir = 'CostGamma/LogRegWt7p5_C1/results'
#dir = 'CostGamma/LogRegWt7p5_Cp1/results'
#dir = 'CostGamma/LogRegWt7p5_C10/results'

#dir = 'FineTune/LogRegCls1_Wt1/results'
#dir = 'FineTune/LogRegCls1_Wtp5/results'
#dir = 'FineTune/LogRegCls1_Wt1p5/results'
#dir = 'FineTune/LogRegCls1_Wt2/results'
#dir = 'FineTune/LogRegCls1_Wt3/results'
#dir = 'FineTune/LogRegCls1_Wt5/results'

#dir = 'FineTune/LogRegCls2_Wt1/results'
#dir = 'FineTune/LogRegCls2_Wtp5/results'
#dir = 'FineTune/LogRegCls2_Wt1p5/results'

#dir = 'FineTune/LogRegCls3_Wt1/results'
#dir = 'FineTune/LogRegCls3_Wtp5/results'
#dir = 'FineTune/LogRegCls3_Wt1p5/results'

#dir = 'FineTune/LogRegCls4_Wt1/results'
#dir = 'FineTune/LogRegCls4_Wtp5/results'
#dir = 'FineTune/LogRegCls4_Wt1p5/results'
#dir = 'FineTune/LogRegCls4_Wt2/results'

#dir = 'FineTune/LogRegCls5_Wt1/results'
#dir = 'FineTune/LogRegCls5_Wtp5/results'
#dir = 'FineTune/LogRegCls5_Wt1p5/results'
#dir = 'FineTune/LogRegCls5_Wt5/results'
#dir = 'FineTune/LogRegCls5_Wt10/results'
#dir = 'FineTune/LogRegCls5_Wt20/results'

#dir = 'FineTune/LogRegCls6_Wt1/results'
#dir = 'FineTune/LogRegCls6_Wtp5/results'
#dir = 'FineTune/LogRegCls6_Wt1p5/results'
#dir = 'FineTune/LogRegCls6_Wt2/results'
#dir = 'FineTune/LogRegCls6_Wt3/results'

#dir = 'FineTune/LogRegCls7_Wt1/results'
#dir = 'FineTune/LogRegCls7_Wtp5/results'
#dir = 'FineTune/LogRegCls7_Wt1p5/results'
#dir = 'FineTune/LogRegCls7_Wt2/results'
#dir = 'FineTune/LogRegCls7_Wt3/results'
#dir = 'FineTune/LogRegCls7_Wt5/results'

#dir = 'FineTune/LogRegCls8_Wt1/results'
#dir = 'FineTune/LogRegCls8_Wtp5/results'
dir = 'FineTune/LogRegCls8_Wtp75/results'
#dir = 'FineTune/LogRegCls8_Wt1p5/results'
fineCls = 8


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
        for lvl in ['fine','trainCls','testCls']:
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
                        if (not isinstance(rec[ind[0]+1], str)):
                            resultMat[colNum].append('{:.3f}'.format(rec[ind[0]+1]))
                        else:
                            resultMat[colNum].append('{}'.format(rec[ind[0] + 1]))

            if (not isinstance(prFold[0], str)):
                prFold = np.array(prFold)
                #print(prFold)
                resultMat[colNum].append('avg {:.3f}'.format(np.mean(prFold)))
            else:
                resultMat[colNum].append('')
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
        for lvl in ['fine','trainCls','testCls']:
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
for lvl in ['fine']:
    table.append(['title']+[lvl])
    for col,colItem in enumerate(resultMat):
        if(lvl in resultMat[col][0]):
            table.append([resultMat[col][0].replace(lvl+'-','')]+[resultMat[col][-1].replace('avg ','')])

tableCol = 0
for lvl in ['trainCls']:
    table[tableCol].append(lvl+'-'+str(fineCls))
    tableCol += 1
    for col,colItem in enumerate(resultMat):
        if(lvl in resultMat[col][0]):
            table[tableCol].append(resultMat[col][-1].replace('avg ',''))
            tableCol += 1

tableCol = 0
for lvl in ['testCls']:
    table[tableCol].append(lvl+'-'+str(fineCls))
    tableCol += 1
    for col,colItem in enumerate(resultMat):
        if(lvl in resultMat[col][0]):
            table[tableCol].append(resultMat[col][-1].replace('avg ',''))
            tableCol += 1
print(table)




tmpCol = ['conf (tn/fn)','','','','','','']
for col,colItem in enumerate(resultConf):
    if('fine-tn' in resultConf[col][0]):
        tmpCol[1] = resultConf[col][-1].replace('avg ','')
    if ('fine-fn' in resultConf[col][0]):
        tmpCol[2] = resultConf[col][-1].replace('avg ', '')
    if('trainCls-tn' in resultConf[col][0]):
        tmpCol[3] = resultConf[col][-1].replace('avg ','')
    if ('trainCls-fn' in resultConf[col][0]):
        tmpCol[4] = resultConf[col][-1].replace('avg ', '')
    if('testCls-tn' in resultConf[col][0]):
        tmpCol[5] = resultConf[col][-1].replace('avg ','')
    if ('testCls-fn' in resultConf[col][0]):
        tmpCol[6] = resultConf[col][-1].replace('avg ', '')
table.append([tmpCol[0]]+['( {} / {} )'.format(tmpCol[1],tmpCol[2])]+
             ['( {} / {} )'.format(tmpCol[3], tmpCol[4])]+
             ['( {} / {} )'.format(tmpCol[5], tmpCol[6])])

tmpCol = ['conf (fp/tp)','','','','','','']
for col,colItem in enumerate(resultConf):
    if('fine-fp' in resultConf[col][0]):
        tmpCol[1] = resultConf[col][-1].replace('avg ','')
    if ('fine-tp' in resultConf[col][0]):
        tmpCol[2] = resultConf[col][-1].replace('avg ', '')
    if('trainCls-fp' in resultConf[col][0]):
        tmpCol[3] = resultConf[col][-1].replace('avg ','')
    if ('trainCls-tp' in resultConf[col][0]):
        tmpCol[4] = resultConf[col][-1].replace('avg ', '')
    if('testCls-fp' in resultConf[col][0]):
        tmpCol[5] = resultConf[col][-1].replace('avg ','')
    if ('testCls-tp' in resultConf[col][0]):
        tmpCol[6] = resultConf[col][-1].replace('avg ', '')
table.append([tmpCol[0]]+['( {} / {} )'.format(tmpCol[1],tmpCol[2])]+
             ['( {} / {} )'.format(tmpCol[3], tmpCol[4])]+
             ['( {} / {} )'.format(tmpCol[5], tmpCol[6])])

print(table)
printResultMat(f,table)
f.close()
