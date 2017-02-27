import numpy as np
import methodsPsv as m


resultMat = []
colNum = 0
resultMat.append([])

resultMat[colNum].append('Titles')
resultMat[colNum].append('fineErr')
resultMat[colNum].append('coarseErr')
resultMat[colNum].append('total')
resultMat[colNum].append('jaccardInd')

resultMat[colNum].append('InterSect')
for i in range(9):
    resultMat[colNum].append(str(i))
resultMat[colNum].append('Total')

resultMat[colNum].append('LogRegCoarse')
for i in range(9):
    resultMat[colNum].append(str(i))
resultMat[colNum].append('Total')

resultMat[colNum].append('SVMFine')
for i in range(9):
    resultMat[colNum].append(str(i))
resultMat[colNum].append('Total')

colNum+=1

clfType = 'SVM' #LogReg, SVM
for fold in [1,2,3,4,5,6,7,8,9,10]:
    resultMat.append([])
    classes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    coarse_cls = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    fine_cls = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    coarseErr = []
    clfType = 'LogReg'
    with open('FmtResultsLogReg/jaccard/'+clfType+'_coarse_'+str(fold)+'.txt') as f:  #FmtResultsSVM, FmtResultsLogReg
    #with open('FmtResultsLogReg/jaccard/' + clfType + '_fine_' + str(fold) + '.txt') as f:  # FmtResultsSVM, FmtResultsLogReg
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            coarseErr.append(nums)

    fineErr = []
    clfType = 'SVM'
    #with open('FmtResultsSVM/jaccard/'+clfType+'_fine_'+str(fold)+'.txt') as f:
    #with open('FmtResultsSVM/jaccard/' + clfType + '_coarse_' + str(fold) + '.txt') as f:  # FmtResultsSVM, FmtResultsLogReg
    with open('FmtResultsSVM/jaccard/' + clfType + '_fine_' + str(fold) + '.txt') as f:  # FmtResultsSVM, FmtResultsLogReg
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            fineErr.append(nums)




    results = []

    #err_file = open('FmtResultsSVM/jaccard/_jaccardTmp_'+str(fold)+'.txt', 'w')
    #err_file = open('_jaccardSvmLogRegCoarse_' + str(fold) + '.txt', 'w')

    interTot = 0
    for coarseInst in coarseErr:
        for fineInst in fineErr:
            if (coarseInst[1:] == fineInst[1:]):
                interTot += 1
                #m.printDataInstance(coarseInst, err_file)
                classes[coarseInst[0]].append(coarseInst)
                # m.printDataInstance(fineInst, err_file)

    total = len(coarseErr) + len(fineErr)
    jaccardInd = interTot / (total - interTot)
    m.addPrint(results, '{} & {} \\\\'.format('fineErr',len(fineErr))) #, 'coarseErr', 'total', 'inter'))
    m.addPrint(results, '{} & {} \\\\'.format('coarseErr', len(coarseErr)))
    m.addPrint(results, '{} & {} \\\\'.format('total', total))
    m.addPrint(results, '{} & {:.2f} \\\\'.format('jaccardInd', jaccardInd))
    resultMat[colNum].append('fld'+str(colNum))
    resultMat[colNum].append(len(fineErr))
    resultMat[colNum].append(len(coarseErr))
    resultMat[colNum].append(total)
    resultMat[colNum].append(round(jaccardInd,2))



    print('{} & {} \\\\'.format('Classes', 'Count'))
    resultMat[colNum].append('')
    instanceCount = 0
    for i in sorted(classes):
        instanceCount += len(classes[i])
        m.addPrint(results,'{} & {} \\\\'.format(i, len(classes[i])))
        resultMat[colNum].append(len(classes[i]))
    m.addPrint(results,'{} & {} \\\\ \n'.format('Total', instanceCount))
    resultMat[colNum].append(instanceCount)

    for coarseInst in coarseErr:
        coarse_cls[coarseInst[0]].append(coarseInst)
    m.addPrint(results,'{} & {} \\\\'.format('Coarse', 'Count'))
    resultMat[colNum].append('')
    instanceCount = 0
    for i in sorted(coarse_cls):
        instanceCount += len(coarse_cls[i])
        m.addPrint(results,'{} & {} \\\\'.format(i, len(coarse_cls[i])))
        resultMat[colNum].append(len(coarse_cls[i]))
    m.addPrint(results,'{} & {} \\\\ \n'.format('Total', instanceCount))
    resultMat[colNum].append(instanceCount)

    for fineInst in fineErr:
        fine_cls[fineInst[0]].append(fineInst)
    m.addPrint(results,'{} & {} \\\\'.format('Fine', 'Count'))
    resultMat[colNum].append('')
    instanceCount = 0
    for i in sorted(fine_cls):
        instanceCount += len(fine_cls[i])
        m.addPrint(results,'{} & {} \\\\'.format(i, len(fine_cls[i])))
        resultMat[colNum].append(len(fine_cls[i]))
    m.addPrint(results,'{} & {} \\\\ \n'.format('Total', instanceCount))
    resultMat[colNum].append(instanceCount)

    # for rec in results:
    #     err_file.write(str(rec) + '\n')
    # err_file.close()
    colNum += 1



avgList = []
for i in range(len(resultMat[0])):
    if(type(resultMat[1][i]) != str):
        avg = []
        for j in range(1,len(resultMat)):
            #print(resultMat[j][i])
            avg.append(resultMat[j][i])
        mean = np.mean(np.array(avg))
        if mean < 1:
            avgList.append('{:.2f}'.format(mean))
        else:
            avgList.append('{:.1f}'.format(mean))
    else:
        if(i==0):
            avgList.append('Avg')
        else:
            avgList.append('')
resultMat.append(avgList)

#f = open('FmtResultsSVM/jaccard/_jaccardTmp.txt', 'w')
f = open('_jaccardSvmFineLogRegCoarse_.txt', 'w')
for i in range(len(resultMat)):
    f.write('|l|')
f.write('\n')
for i, row in enumerate(resultMat[0]):
    for j, col in enumerate(resultMat[:-1]):
        f.write('{} & '.format(resultMat[j][i]))
    if(i == 0):
        f.write('{} \\\\ \\midrule'.format(resultMat[-1][i]))
    elif(i== (len(resultMat[0])-1)):
        f.write('{} \\\\ \\bottomrule'.format(resultMat[-1][i]))
    else:
        f.write('{} \\\\'.format(resultMat[-1][i]))
    f.write('\n')
f.close()