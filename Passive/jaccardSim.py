import numpy as np
import methods as m

classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}

coarseErr = []
with open("other_results/_minDimPartlib8DecFcn_coarseErr.txt") as f:
#with open("other_results/_sclSelPartlib8DecFcn_coarseErr.txt") as f:
#with open("other_results/_sclSelPartlib8DecFcn_fineErr.txt") as f:
    for line in f:
        nums = line.split()
        nums = list(map(float, nums))
        coarseErr.append(nums)

fineErr = []
with open("other_results/_minDimPartlib8DecFcn_fineErr.txt") as f:
#with open("other_results/_sclSelPartlib8DecFcn_fineErr.txt") as f:
#with open("other_results/_minDimPartlib8DecFcn_coarseErr.txt") as f:
    for line in f:
        nums = line.split()
        nums = list(map(float, nums))
        fineErr.append(nums)

err_file = open('other_results/_jaccardTmp.txt', 'w')
interTot = 0
for coarseInst in coarseErr:
    for fineInst in fineErr:
        if(coarseInst[1:] == fineInst[1:]):
            interTot +=1
            m.printDataInstance(coarseInst, err_file)
            classes[coarseInst[0]].append(coarseInst)
            #m.printDataInstance(fineInst, err_file)
err_file.close()

total = len(coarseErr)+len(fineErr)
jaccardInd = interTot / (total-interTot)
print('{:<7}{:<7}{:<7}{:<7}'.format('log','svm','total','inter'))
print('{:<7}{:<7}{:<7}{:<7}'.format(len(fineErr),len(coarseErr),total,interTot))
print('{:<7}{:<7}{:<7.3f}'.format('jaccard','=',jaccardInd))




print('{0:<10}{1:<10}'.format('Classes', ''))
instanceCount = 0
for i in sorted(classes):
    instanceCount += len(classes[i])
    print('{0:<10}{1:<10}'.format(i, len(classes[i])))
print('{0:<10}{1:<10}\n'.format('Total', instanceCount))

