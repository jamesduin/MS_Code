import numpy as np


#partitions = [ [],[],[],[],[],[],[],[],[],[] ]
coarse_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
fine_set = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
classes_all = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}



#### store totals
# totals = []
# for i in sorted(classes_all):
#     classes_all[i] = np.loadtxt("../data/classes/class_" + str(i))
#     np.random.shuffle(classes_all[i])
#     totals.append(len(classes_all[i]))
# tot = np.array(totals)
# totVect = tot/np.sum(tot)

totals = []
for i in sorted(classes_all):
    with open("../data/classes_part/class_" + str(i)) as f:
        for line in f:
            classes_all[i].append(line.split())
            #print(line.split())
    #classes_all[i] = np.loadtxt("../data/classes/class_" + str(i))
#     np.random.shuffle(classes_all[i])
#     totals.append(len(classes_all[i]))
# tot = np.array(totals)
# totVect = tot/np.sum(tot)




#
# print("//////////////// Classes ////////////////")
# instanceCount = 0
# for i in sorted(classes_all):
#     instanceCount += len(classes_all[i])
#     print(str(i)+","+str(len(classes_all[i])) )
# print( "Total => " + str(instanceCount) )
#
#
#
# #### randomly add 16 to starter coarse set
# coarseStart = np.ceil(totVect*10)
# for i in sorted(classes_all):
#     for j in range(int(coarseStart[i])):
#         coarse_set[i].append(classes_all[i][j])
# for i in sorted(coarse_set):
#     print(str(i) + "," + str(len(coarse_set[i])))
#
#
#
# #### randomly add 44 to starter fine set
# fineStart = np.ceil(totVect*40)
# for i in sorted(classes_all):
#     for j in range(int(fineStart[i])):
#         fine_set[i].append(classes_all[i][j])
# for i in sorted(fine_set):
#     print(str(i) + "," + str(len(fine_set[i])))
#
#
#
# print("//////////////// Classes ////////////////")
# instanceCount = 0
# for i in sorted(classes_all):
#     instanceCount += len(classes_all[i])
#     print(str(i)+","+str(len(classes_all[i])) )
# print( "Total => " + str(instanceCount) )
