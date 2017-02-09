import numpy as np
from sklearn.externals import joblib
from sklearn import linear_model
import time
import re
import methodsPsvOOP as m

fName = re.split("[/\.]",__file__)[-2]
lvl = re.split("_",fName)[1]


classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
coarse_folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
results_coarse = []




# load the data
for i in sorted(coarse_folds):
    with open("../data/partition/partition_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            coarse_folds[i].append(nums)
    np.random.shuffle(coarse_folds[i])
start_time = time.perf_counter()






##### iterate through fold list for coarse
# fold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# #fold_list = [1]
# for testFold in fold_list:

testFold = 1
f = open('results/_'+fName+'.txt', 'w')
print(lvl+" fold" + str(testFold))
f.write('{} fold {}\n'.format(lvl,testFold))

##### create train set and train_wt for coarse
y_train, X_train, min_max_scaler = m.createTrainSet(coarse_folds,testFold)

##### create train_wt and y_train for coarse
y_trainCoarse = []
for i in y_train:
    if i > 0:
        y_trainCoarse.append(1.)
    else:
        y_trainCoarse.append(0.)
train_wt = m.fcnSclWeight(len(y_train)/np.sum(y_trainCoarse))



##### create test set and sample weight
y_testCoarse,X_test,y_sampleWeight = m.createTestSet(coarse_folds,testFold, min_max_scaler,train_wt)




##### train classifier for coarse
classifier = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.1,
                                          fit_intercept=False, intercept_scaling=1,
                                          class_weight={1: train_wt},
                                          solver='liblinear',
                                          max_iter=1000, n_jobs=-1)
clf = classifier.fit(X_train,y_trainCoarse)
joblib.dump(clf,lvl+'_models/'+fName+'_'+str(testFold)+'.pkl')




##### predict test set for coarse
y_predCoarse = clf.predict(X_test)
y_score = clf.decision_function(X_test)

###### print conf matrix,accuracy and f1_score
m.printConfMatrix(y_testCoarse, y_predCoarse,f)


###### Plot ROC and PR curves
m.plotRocPrCurves(y_testCoarse,y_score,y_sampleWeight,lvl,fName,testFold,results_coarse)

###### Save results to a file
m.saveResults(lvl,results_coarse,f,start_time)







