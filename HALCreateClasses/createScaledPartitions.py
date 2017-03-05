from sklearn import preprocessing
import numpy as np
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import os
import re
import shutil


def printDataInstance(instance):
    for dimension in instance[:-1]:
        f.write(str(dimension) + " ")
    f.write(str(instance[-1]))
    f.write("\n")
    return




dir = '../data/part_subSel75'
if not os.path.exists(dir):
    os.makedirs(dir)

fname = re.split('[/]',dir)[2]

classes_PreScale = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
#### load Data
for i in sorted(classes_PreScale):
    with open('../data/classes_subset/class_' + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes_PreScale[i].append(nums)
data_PreScale = []
for i in sorted(classes_PreScale):
    partition = np.asarray(classes_PreScale[i])
    if len(data_PreScale) == 0:
        data_PreScale = partition
    else:
        data_PreScale = np.vstack((partition, data_PreScale))
y_train, X_trainPreScale = data_PreScale[:, 0], data_PreScale[:, 1:data_PreScale.shape[1]]

#
#### Scale dataset
# normalizer = preprocessing.Normalizer().fit(X_trainPreScale)
# X_train = normalizer.transform(X_trainPreScale)


# min_max_scaler = preprocessing.MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_trainPreScale)


scaler = preprocessing.StandardScaler().fit(X_trainPreScale)
X_trainFull = scaler.transform(X_trainPreScale)
#X_train = scaler.transform(X_trainPreScale)
selector = SelectPercentile(f_classif, percentile=75)
selector.fit(X_trainFull, y_train)
X_train = selector.transform(X_trainFull)

y_train = np.reshape(y_train, (y_train.shape[0], 1))

data = np.hstack((y_train, X_train))
for inst in data:
    classes_all[inst[0]].append(inst)

all_part = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
for i in sorted(classes_all):
    np.random.shuffle(classes_all[i])
    partList = []
    for j in sorted(all_part):
        partList.append((j, len(all_part[j])))
    minIndex = partList[0][0]
    minVal = partList[0][1]
    for j in sorted(partList):
        if (minVal > j[1]):
            minVal = j[1]
            minIndex = j[0]
    partitionCounter = minIndex
    instCount = 1
    for instance in classes_all[i]:
        #if not (i == 0 and instCount < 3827):
        instCount += 1
        all_part[partitionCounter].append(instance)
        partitionCounter += 1
        if partitionCounter > 10:
            partitionCounter = 1
for i in sorted(all_part):
    np.random.shuffle(all_part[i])





stdout = open('../data/'+fname+'/terminalout.txt', 'w')

stdout.write('{} & {} \\\\ \n'.format('Classes', ''))
print('{} & {} \\\\'.format('Classes', ''))
instanceCount = 0
for i in sorted(classes_all):
    instanceCount += len(classes_all[i])
    stdout.write('{} & {} \\\\ \n'.format(i, len(classes_all[i])))
    print('{} & {} \\\\'.format(i, len(classes_all[i])))
stdout.write('{} & {} \\\\ \n'.format('Total', instanceCount))
print('{} & {} \\\\ \n'.format('Total', instanceCount))
stdout.write('{} & {} \\\\ \n'.format('Shape',len(classes_all[0][0])))
print('{} & {} \\\\'.format('Shape',len(classes_all[0][0])))


stdout.write('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format('part','Folds',0,1,2,3,4,5,6,7,8))
print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format('part','Folds',0,1,2,3,4,5,6,7,8))
instanceCount = 0
classCountTot = [0,0,0,0,0,0,0,0,0]
for i in sorted(all_part):
    classCount = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    instanceCount += len(all_part[i])
    for inst in all_part[i]:
        classCountTot[int(inst[0])]+=1
        classCount[int(inst[0])] += 1
    classCount = [i] + [len(all_part[i])] + classCount
    stdout.write('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(*classCount))
    print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(*classCount))
stdout.write('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format('Total',instanceCount,*classCountTot))
print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format('Total',instanceCount,*classCountTot))






for eachFold in sorted(all_part):
    np.random.shuffle(all_part[eachFold])
    f = open('../data/'+fname+'/'+fname+'_' + str(eachFold), 'w')
    count = 0
    for instance in all_part[eachFold]:
        printDataInstance(instance)
        count += 1
    stdout.write('class,count: {:<5}{:<5}\n'.format(eachFold,count))
    print('class,count: {:<5}{:<5}'.format(eachFold, count))
    f.close()

stdout.close()


