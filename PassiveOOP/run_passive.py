import numpy as np
import time
import re
import methodsPsvOOP as m

classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
partitions = ['1_10','1_5','1_2','1_1']
#partitions = ['scaled']
#partitions = ['1_1']
results = dict()

for part in partitions:
    # load the data
    folds = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for i in sorted(folds):
        with open('../data/partition_'+part+'/partition_'+part+'_'+ str(i)) as f:
            for line in f:
                nums = line.split()
                nums = list(map(float, nums))
                folds[i].append(nums)
        np.random.shuffle(folds[i])

    #fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fold_list = [2]

    rnds = dict()
    rnds['coarse'] = m.CoarseRound(folds,'passive_coarse_'+part)
    rnds['fine'] = m.FineRound(folds,'passive_fine_'+part)
    results[part] = dict()
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

        #rnds[lvl].saveResults(start_time)
        results[part][lvl] =rnds[lvl].results

    #m.printClsVsFolds(rnds['fine'].folds, 'folds')



print('{0:38}{1:33}'.format('coarse','fine'))
print('{0:5}'
      '{1:5}{2:7}{3:7}{4:7}{5:7}'
      '{1:5}{2:7}{3:7}{4:7}{5:7}'.format('fold','dec', 'pr', 'roc','acc','f1'))
for part in sorted(results):
    for i in range(len(results[part]['fine'])):
        if(results[part]['coarse'][i][0] == ' '):
            print('{:<5}'
                  '{:<5}{:<7}{:<7}{:<7}{:<7}'
                  '{:<5}{:<7}{:<7}{:<7}{:<7}'.format(part,*results[part]['coarse'][i],
                                                    *results[part]['fine'][i]))
        else:
            print('{:<5}'
                  '{:<5}{:<7.3f}{:<7.3f}{:<7.3f}{:<7.3f}'
                  '{:<5}{:<7.3f}{:<7.3f}{:<7.3f}{:<7.3f}'.format(part,*results[part]['coarse'][i],
                                                    *results[part]['fine'][i]))

print('{} sec'.format(round(time.perf_counter() - start_time, 2)))
