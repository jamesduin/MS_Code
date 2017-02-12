import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import re



plt.figure()
#with plt.style.context('fivethirtyeight'):
plt.style.use('ggplot')



rndTypeSet = set()
for fname in glob.glob('results/*.res'):
    #print(fname)
    file = re.split("[/\.]", fname)[1]
    rndType = re.split("[_]", file)
    #print(rndType[0]+'_'+rndType[1])
    rndTypeSet.add(rndType[0]+'_'+rndType[1])
print(rndTypeSet)
#rndTypeSet = {'passive_coarse', 'active_fine', 'active_coarse', 'passive_fine'}
#rndTypeSet = {'passive_coarse'}

for type in rndTypeSet:
    prs = []
    foldMatrix = dict()
    fold = 0
    for fname in glob.glob('results/*.res'):
        file = re.split("[/\.]", fname)[1]
        rndType = re.split("[_]", file)
        instType = rndType[0]+'_'+rndType[1]
        if(type == instType):
            fold +=1
            pr_row = []
            foldMatrix[fold] = []
            results = pickle.load(open(fname, 'rb'))
            for result in results:
                foldMatrix[fold].append(result)
                fnd = False
                for item in result:
                    if (fnd):
                        pr_row.append(item)
                        fnd = False
                    elif item == 'pr':
                        fnd = True
            if prs == []:
                prs = np.array(pr_row).reshape(len(pr_row),1)
            else:
                size = prs.shape[0]
                print(size)
                row = pr_row[:size]
                while(len(row)!= size):
                    row.append([0])
                pr = np.array(row).reshape(len(row),1)
                prs = np.hstack((prs,pr))
    f = open('results/_' + type + '.txt', 'w')
    for i in range(len(foldMatrix[1])):
        for fold in foldMatrix:
            f.write(str(foldMatrix[fold][i])+'\n')

    # print((prs[0][0]+prs[0][1]+prs[0][2])/3)
    x = np.array(range(1,len(prs)+1))
    y = np.mean(prs, axis=1)
    print(x)
    print(y)
    plt.plot(x,y, label = 'avg_'+type)

plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('Active vs. Passive Learning')
plt.legend(loc="lower right")
plt.savefig('results/ActiveVsPassive.png')


# results = pickle.load(open(fname, 'rb'))
# x = []
# y = []
# for result in results:
#     if(result[0] == count):
#         x.append(result[0])
#         y.append(result[3])
# plt.plot(x,y, label = fname)
#
# plt.ylabel('PR-AUC')
# plt.xlabel('Iteration')
# plt.title('Active vs. Passive Learning')
# plt.legend(loc="lower right")
# plt.savefig('results/ActiveVsPassive.png')
#
#
# for fname in glob.glob('results/*.res'):
#     results = pickle.load(open(fname, 'rb'))
#     outName = re.split("[/\.]", fname)[1]
#     # ###### Save coarse results to a file
#     f = open('results/_' + outName + '.txt', 'w')
#     for result in results:
#         f.write(str(result) + '\n')
#     f.close()


#
# #lvls = ['passive','active']
# lvls = ['passive']
# for lvl in lvls:
#     fileName1 = open('results/'+lvl+'_fine.res','rb')
#     fine_rnds = pickle.load(fileName1)
#     fileName1.close()
#
#     fileName2 = open('results/'+lvl+'_coarse.res','rb')
#     coarse_rnds = pickle.load(fileName2)
#     fileName2.close()
#
#     f = open('results/_'+lvl+'_rnds.txt', 'w')
#     results = []
#     f.write('{:^20}       {:^20}\n'.format('coarse','fine'))
#     for i,result in enumerate(coarse_rnds):
#         if(type(result[0]) != str and coarse_rnds[i][0] == fine_rnds[i][0]):
#             f.write('{0:<4}{1:<5.3f},{2:<5.3f}        {3:<4}{4:<5.3f},{5:<5.3f}\n'.format(coarse_rnds[
#                                                                                                                   i][0],
#                                                                                                               coarse_rnds[
#                                                                                                                   i][1],
#                                                                                                               coarse_rnds[
#                                                                                                                   i][2],
#                                                                                                               fine_rnds[i][
#                                                                                                                   0],
#                                                                                                               fine_rnds[i][
#                                                                                                                   1],
#                                                                                                               fine_rnds[i][
#                                                                                                                   2]))
#     f.close()
