import numpy as np
import fileinput

partitions = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}


# data = []
# partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for x in partition_list:
#     partition = np.loadtxt("../data/classes_subset/class_" + str(x))
#     if data == []:
#         data = partition
#     else:
#         data = np.vstack((partition, data))

for eachClass in sorted(classes):
    classes[eachClass] = np.loadtxt("../data/classes_subset/class_" + str(eachClass))
    np.random.shuffle(classes[eachClass])
    partitionCounter = int(np.floor(np.random.random_sample()*10)+1)
    for instance in classes[eachClass]:
        partitions[partitionCounter].append(instance)
        partitionCounter+=1
        if partitionCounter > 10:
            partitionCounter = 1


print("//////////////// PARTITIONS ////////////////")
instanceCount = 0
for eachPartition in sorted(partitions):
    instanceCount += len(partitions[eachPartition])
    print(str(eachPartition)+","+str(len(partitions[eachPartition])) )
print( "Total => " + str(instanceCount) )
