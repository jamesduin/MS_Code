import numpy as np
import methodsPsv as m
import re

clfType = 'SVM' #LogReg, SVM
PsvResMat = dict()
for fold in [1]:#[1,2,3,4,5,6,7,8,9,10]:
    PsvResMat[fold] = []
    with open( 'results'+clfType+'/results/Psv_' +str(fold ) +'.txt') as f:
        for line in f:
            nums = line
            PsvResMat[fold].append(nums)


confCoarse = dict()

for fold in PsvResMat:
    for rec in PsvResMat[fold]:
        #print(rec)
        if ('conf' in rec and 'combPred' not in rec
            and 'coarse' in rec):
            print(rec[:-1])
            arr = rec[:-1].replace(']','').replace('[','').replace('\'','').replace(' ','').split(',')
            print(arr)
            if('tn' in rec):
                confCoarse['tn'] = int(arr[-3])