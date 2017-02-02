import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import time



classes_subset = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}



def printDataInstance(instance):
    for dimension in instance[:-1]:
        f.write(str(dimension) + " ")
    f.write(str(instance[-1]))
    f.write("\n")
    return



#### store totals
totals = []
for i in sorted(classes_subset):
    with open("../data/classes_subset/class_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes_subset[i].append(nums)
    np.random.shuffle(classes_subset[i])
    totals.append(len(classes_subset[i]))
tot = np.array(totals)
totVect = tot/np.sum(tot)



##### Create folds for coarse set
for i in sorted(classes_subset):
    np.random.shuffle(classes_subset[i])
    partList = []
    for j in sorted(folds):
        partList.append((j,len(folds[j])))
    minIndex = partList[0][0]
    minVal = partList[0][1]
    for j in sorted(partList):
        if(minVal > j[1] ):
            minVal = j[1]
            minIndex = j[0]
    partitionCounter = minIndex
    for instance in classes_subset[i]:
        folds[partitionCounter].append(instance)
        partitionCounter+=1
        if partitionCounter > 10:
            partitionCounter = 1

for i in sorted(folds):
    np.random.shuffle(folds[i])




print('{0:<10}{1:<10}'.format('Classes', ''))
instanceCount = 0
for i in sorted(classes_subset):
    instanceCount += len(classes_subset[i])
    print('{0:<10}{1:<10}'.format(i, len(classes_subset[i])))
print('{0:<10}{1:<10}\n'.format('Total', instanceCount))



print('{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Fine','Folds',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(folds):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(folds[i])
    for inst in folds[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(folds[i])] + classCount
    print('{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format(*classCount))

print('{:<7}{:<7}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}{:<5}'.format('Total',instanceCount,*classCountTot))






for eachFold in sorted(folds):
    np.random.shuffle(folds[eachFold])
    f = open('partition_subset/partition_sub' + str(eachFold), 'w')
    count = 0
    for instance in folds[eachFold]:
        printDataInstance(instance)
        count += 1
    print("class,count: %s,%s" % (eachFold,count))
    f.close()












