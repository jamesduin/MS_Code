import numpy as np
import fileinput
from sklearn import preprocessing

partitions = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
classes_PreScale = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}


def printDataInstance(instance):
    for dimension in instance[:-1]:
        f.write(str(dimension) + " ")
    f.write(str(instance[-1]))
    f.write("\n")
    return





for i in sorted(classes_PreScale):
    with open("classes/class_" + str(i)) as f:
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

#### Scale dataset
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_trainPreScale)
y_train = np.reshape(y_train,(y_train.shape[0],1))

data = np.hstack((y_train,X_train))

for inst in data:
    classes[inst[0]].append(inst)



print('{0:<10}{1:<10}'.format('Classes', ''))
instanceCount = 0
for i in sorted(classes_PreScale):
    instanceCount += len(classes_PreScale[i])
    print('{0:<10}{1:<10}'.format(i, len(classes_PreScale[i])))
print('{0:<10}{1:<10}\n'.format('Total', instanceCount))




for eachClass in sorted(classes):
    np.random.shuffle(classes[eachClass])
    f = open('classes_scaled/class_scaled' + str(eachClass), 'w')
    count = 1
    for instance in classes[eachClass]:
        printDataInstance(instance)
        count += 1
    print("class,count: %s,%s" % (eachClass,count))
    f.close()

