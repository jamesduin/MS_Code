import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import re



plt.figure()
#with plt.style.context('fivethirtyeight'):
plt.style.use('ggplot')

for fname in glob.glob('results/*.res'):
    print(fname)
    results = pickle.load(open(fname, 'rb'))
    x = []
    y = []
    count = 1
    for result in results:
        if(result[0] == count):
            x.append(result[0])
            y.append(result[2])
            count+=1
    plt.plot(x,y, label = fname)

plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('Active vs. Passive Learning')
plt.legend(loc="lower right")
plt.savefig('results/ActiveVsPassive.png')


for fname in glob.glob('results/*.res'):
    results = pickle.load(open(fname, 'rb'))
    outName = re.split("[/\.]", fname)[1]
    # ###### Save coarse results to a file
    f = open('results/_' + outName + '.txt', 'w')
    for result in results:
        f.write(str(result) + '\n')
    f.close()


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
