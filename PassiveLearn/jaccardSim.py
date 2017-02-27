import numpy as np
import methodsPsv as m



clfType = 'SVM' #LogReg, SVM
for fold in [1,2,3,4,5,6,7,8,9,10]:
    classes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    coarse_cls = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    fine_cls = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    coarseErr = []
    with open('resultsSVM/jaccard/'+clfType+'_coarse_'+str(fold)+'.txt') as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            coarseErr.append(nums)

    fineErr = []
    with open('resultsSVM/jaccard/'+clfType+'_fine_'+str(fold)+'.txt') as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            fineErr.append(nums)




    results = []

    err_file = open('resultsSVM/jaccard/_jaccardTmp_'+str(fold)+'.txt', 'w')

    interTot = 0
    for coarseInst in coarseErr:
        for fineInst in fineErr:
            if (coarseInst[1:] == fineInst[1:]):
                interTot += 1
                m.printDataInstance(coarseInst, err_file)
                classes[coarseInst[0]].append(coarseInst)
                # m.printDataInstance(fineInst, err_file)

    total = len(coarseErr) + len(fineErr)
    jaccardInd = interTot / (total - interTot)
    m.addPrint(results,'{} & {} & {} & {} \\\\'.format('fineErr', 'coarseErr', 'total', 'inter'))
    m.addPrint(results,'{} & {} & {} & {} \\\\'.format(len(fineErr), len(coarseErr), total, interTot))
    m.addPrint(results,'{} & {} & {} & {:.3f} \\\\'.format('jaccard','ind', '=', jaccardInd))

    print('{} & {} \\\\'.format('Classes', 'Count'))
    instanceCount = 0
    for i in sorted(classes):
        instanceCount += len(classes[i])
        m.addPrint(results,'{} & {} \\\\'.format(i, len(classes[i])))
    m.addPrint(results,'{} & {} \\\\ \n'.format('Total', instanceCount))

    for coarseInst in coarseErr:
        coarse_cls[coarseInst[0]].append(coarseInst)
    m.addPrint(results,'{} & {} \\\\'.format('Coarse', 'Count'))
    instanceCount = 0
    for i in sorted(coarse_cls):
        instanceCount += len(coarse_cls[i])
        m.addPrint(results,'{} & {} \\\\'.format(i, len(coarse_cls[i])))
    m.addPrint(results,'{} & {} \\\\ \n'.format('Total', instanceCount))

    for fineInst in fineErr:
        fine_cls[fineInst[0]].append(fineInst)
    m.addPrint(results,'{} & {} \\\\'.format('Fine', 'Count'))
    instanceCount = 0
    for i in sorted(fine_cls):
        instanceCount += len(fine_cls[i])
        m.addPrint(results,'{} & {} \\\\'.format(i, len(fine_cls[i])))
    m.addPrint(results,'{} & {} \\\\ \n'.format('Total', instanceCount))


    for rec in results:
        err_file.write(str(rec) + '\n')
    err_file.close()