import numpy as np
import time
import re
import methodsPsvOOP as m

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

# load the data
for i in sorted(coarse_folds):
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            coarse_folds[i].append(nums)
            fine_folds[i].append(nums)
    np.random.shuffle(coarse_folds[i])
    np.random.shuffle(fine_folds[i])


fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#fold_list = [1]

rnds = dict()
rnds['coarse'] = m.CoarseRound(coarse_folds,'passive_coarseOOP')
rnds['fine'] = m.FineRound(coarse_folds,'passive_fineOOP')

for lvl in rnds:
    start_time = time.perf_counter()
    for testFold in fold_list:
        rnds[lvl].testFold = testFold
        rnds[lvl].createTrainSet()
        rnds[lvl].createTrainWtYtrain()
        rnds[lvl].createTestSet()
        rnds[lvl].trainClassifier()
        rnds[lvl].predictTestSet()
        rnds[lvl].printConfMatrix()
        rnds[lvl].plotRocPrCurves()
    rnds[lvl].saveResults(start_time)


