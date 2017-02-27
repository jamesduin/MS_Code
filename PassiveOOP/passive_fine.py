import numpy as np
import time
import re
import methodsPsvOOP as m

fName = re.split("[/\.]",__file__)[-2]
lvl = re.split("_",fName)[1]

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
results_fine = []

# load the data
for i in sorted(fine_folds):
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            fine_folds[i].append(nums)
    np.random.shuffle(fine_folds[i])

##### iterate through fold list for fine
# fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# #fold_list = [1]
# for testFold in fold_list:

testFold = 1
start_time = time.perf_counter()
rnd = m.FineRound(lvl,testFold,fine_folds,fName)

rnd.createTrainSet()
rnd.createTrainWtYtrain()
rnd.createTestSet()
rnd.trainClassifier()
rnd.predictTestSet()
rnd.printConfMatrix()
rnd.plotRocPrCurves()
rnd.saveResults(start_time)




