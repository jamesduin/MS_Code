import numpy as np
import fileinput

partitions = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
classesOrigOrder = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1}
classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}

def getClassCode( classCodeStr ):
    if (classCodeStr == "0,0,-1"):
        return 0
    elif (classCodeStr == "1,1,1"):
        return 1
    elif (classCodeStr == "1,0,-1"):
        return 2
    elif (classCodeStr == "1,0,0"):
        return 3
    elif (classCodeStr == "1,0,1"):
        return 4
    elif (classCodeStr == "1,0,2"):
        return 5
    elif (classCodeStr == "1,0,3"):
        return 6
    elif (classCodeStr == "1,0,4"):
        return 7
    elif (classCodeStr == "1,0,5"):
        return 8
    else:
        return "Bad"

def printRawInstance(instance):
    for dimension in instance[:-1]:
        f.write(dimension + ",")
    f.write("\n")
    return

def printDataInstance(instance):
    index = 0
    for dimension in instance[:-1]:
        if index not in [1,2,3,4,5]:
            if dimension == 'N':
                f.write("0 ")
            elif dimension == 'Y':
                f.write("1 ")
            else:
                f.write(dimension + " ")
        index += 1
    f.write(instance[-1])
    f.write("\n")
    return

######################################################################
######################################################################

for line in fileinput.input("../data/mitochondria.tsv"):
    instance = line.split()
    if(len(instance) == 453 and instance[1] != '00mitochondrion'):
        classCodeStr = instance[1] + ',' + instance[2] + ',' + instance[3]
        classCode = getClassCode(classCodeStr)
        classes[classCode].append([str(classCode)]+[str(classesOrigOrder[classCode])]+instance)
        classesOrigOrder[classCode]+=1

for eachClass in sorted(classes):
    np.random.shuffle(classes[eachClass])
    f = open('classes/class_' + str(eachClass), 'w')
    count = 1
    for instance in classes[eachClass]:
        # if eachClass == 0:
        #     if count <= 3827:
        #         printDataInstance(instance)
        #     else:
        #         break
        # else:
        printDataInstance(instance)
        count += 1
    print("class,count: %s,%s" % (eachClass,count))
    f.close()



# partitionStarter = {0:1,1:8,2:1,3:6,4:1,5:10,6:1,7:1,8:5}
#
# for eachClass in sorted(classes_subset):
#     np.random.shuffle(classes_subset[eachClass])
#     if eachClass == 0:
#         partitionCounter = partitionStarter[eachClass]
#         instCount = 1
#         for instance in classes_subset[eachClass]:
#             if instCount <= 3827:
#                 partitions[partitionCounter].append(instance)
#                 instCount += 1
#                 partitionCounter += 1
#                 if partitionCounter > 10:
#                     partitionCounter = 1
#     else:
#         #partitionCounter = int(np.floor(np.random.random_sample()*10)+1)
#         partitionCounter = partitionStarter[eachClass]
#         for instance in classes_subset[eachClass]:
#             partitions[partitionCounter].append(instance)
#             partitionCounter+=1
#             if partitionCounter > 10:
#                 partitionCounter = 1

######################################################################
######################################################################
#
# print("//////////////// PRINT PARTITIONS ////////////////")
# for eachPartition in sorted(partitions):
#     np.random.shuffle(partitions[eachPartition])
#     f = open('partition_'+str(eachPartition), 'w')
#     for instance in partitions[eachPartition]:
#         #printRawInstance(instance)
#         printDataInstance(instance)
#     f.close()
#
# print("//////////////// CLASSES ////////////////")
# instClassCnt = 0
# for eachClass in sorted(classes_subset):
#     instClassCnt += len(classes_subset[eachClass])
#     print(str(eachClass)+","+str(len(classes_subset[eachClass])) )
# print( "Total => " + str(instClassCnt) )
#
# print("//////////////// PARTITIONS ////////////////")
# instanceCount = 0
# for eachPartition in sorted(partitions):
#     instanceCount += len(partitions[eachPartition])
#     print(str(eachPartition)+","+str(len(partitions[eachPartition])) )
# print( "Total => " + str(instanceCount) )








