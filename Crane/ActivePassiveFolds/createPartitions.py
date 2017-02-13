import numpy as np
import methodsActPass as m


classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
folds = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

fname = 'partition_1_1'


def printDataInstance(instance):
    for dimension in instance[:-1]:
        f.write(str(dimension) + " ")
    f.write(str(instance[-1]))
    f.write("\n")
    return



#### store totals
totals = []
for i in sorted(classes):
    with open("../data/classes/class_" + str(i)) as f:
        for line in f:
            nums = line.split()
            nums = list(map(float, nums))
            classes[i].append(nums)
    np.random.shuffle(classes[i])
    totals.append(len(classes[i]))
tot = np.array(totals)
totVect = tot/np.sum(tot)



##### Create folds for coarse set
for i in sorted(classes):
    np.random.shuffle(classes[i])
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
    instCount = 1
    for instance in classes[i]:
        if not (i == 0 and instCount > 10000000000):
            instCount+=1
            folds[partitionCounter].append(instance)
            partitionCounter+=1
            if partitionCounter > 10:
                partitionCounter = 1

for i in sorted(folds):
    np.random.shuffle(folds[i])


stdout = open('../data/'+fname+'/terminalout.txt', 'w')
m.printClassTotals(classes,stdout)
m.printClsVsFolds(folds, fname,stdout)


for eachFold in sorted(folds):
    np.random.shuffle(folds[eachFold])
    f = open('../data/'+fname+'/'+fname+'_' + str(eachFold), 'w')
    count = 0
    for instance in folds[eachFold]:
        printDataInstance(instance)
        count += 1
    stdout.write('class,count: {:<5}{:<5}\n'.format(eachFold,count))
    print('class,count: {:<5}{:<5}'.format(eachFold, count))
    f.close()

stdout.close()










