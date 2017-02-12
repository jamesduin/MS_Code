import numpy as np
import time
import methodsActPass as m
import pickle
import copy

rndTypes = ['passive', 'active']
for rndType in rndTypes:
    testFolds = [1,2,3,4,5,6,7,8,9,10]
    #testFolds = [9]
    for testFold in testFolds:
        start_time = time.perf_counter()

        classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
        train_part = {1:[],2: [],3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        sets = dict()
        sets['coarse'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
        sets['fine'] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
        rnd_results = dict()
        rnd_results['coarse'] = []
        rnd_results['fine'] = []
        classes = dict()

        # using the pre partitioned data
        for i in sorted(train_part):
            with open("../data/partition_1_1/partition_1_1_" + str(i)) as f:
                for line in f:
                    nums = line.split()
                    nums = list(map(float, nums))
                    train_part[i].append(nums)
            np.random.shuffle(train_part[i])
        m.printClsVsFolds(train_part, 'all')
        test_part = train_part[testFold]
        del train_part[testFold]
        m.printClsVsFolds(train_part, 'train')
        classTestTot = m.printClsVsFolds({testFold:test_part}, 'test')
        if(classTestTot[7] == 2):
            ## not enough of class 5 to have 2 in the test set
            m.switchClass5instance(test_part,train_part)
            m.printClsVsFolds(train_part, 'train_mod')
            m.printClsVsFolds({testFold: test_part}, 'test_mod')


        for i in sorted(train_part):
            for index in range(len(train_part[i])):
                classes_all[train_part[i][index][0]].append(train_part[i][index])

        m.printClassTotals(classes_all)

        #### randomly add to starter sets
        start = [952,10,11,16,11,10,10,10,10]
        print(start)
        print("Sum start => "+str(np.sum(start)))
        for i in sorted(classes_all):
            for j in range(int(start[i])):
                inst = classes_all[i].pop()
                for lvl in ['coarse','fine']:
                    sets[lvl][i].append(inst)
        for lvl in ['coarse','fine']:
            classes[lvl] = copy.deepcopy(classes_all)

        rndNum = 1
        #while((18088-instanceCount) > 100):
        while(rndNum < 3):
            if(rndNum>1):
                start_time = time.perf_counter()
                for lvl in ['coarse', 'fine']:
                    if(rndType == 'passive'):
                        m.randAdd(classes[lvl],sets[lvl],100)
                    elif(rndType == 'active'):
                        ###### run confidence estimate for coarse and fine
                        m.confEstAdd(classes[lvl],sets[lvl],rnds[lvl],100)

            rndNum += 1

            rnds = dict()
            rnds['coarse'] = m.CoarseRound(testFold,rndNum)
            rnds['fine'] = m.FineRound(testFold,rndNum)
            #### Run rounds
            for lvl in ['coarse', 'fine']:
                y_train, X_train = rnds[lvl].createTrainSet(sets[lvl])
                y_trainCoarse = rnds[lvl].createTrainWtYtrain(y_train)
                y_testCoarse, y_sampleWeight, X_test = rnds[lvl].createTestSet(test_part)
                rnds[lvl].trainClassifier(X_train, y_trainCoarse)
                y_predCoarse, y_pred_score = rnds[lvl].predictTestSet(X_test)
                rnds[lvl].printConfMatrix(y_testCoarse, y_predCoarse, rnd_results[lvl])
                rnds[lvl].plotRocPrCurves(y_testCoarse, y_pred_score, y_sampleWeight, rnd_results[lvl])
                ##### Append round time and fold counts
                instanceCount = m.appendRndTimesFoldCnts(testFold, rndNum, lvl, rnd_results[lvl], sets[lvl], start_time)
                fileName = open('results/'+rndType+'_'+lvl+'_'+str(testFold)+'.res','wb')
                pickle.dump(rnd_results[lvl],fileName)
                fileName.close()


