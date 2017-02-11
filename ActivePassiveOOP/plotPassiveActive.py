import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob

#glob.glob('plots/*.res')

for fname in glob.glob('plots/*'):
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
#
# fileName = open('results/active_fine','rb')
# ActiveBatches_fine = pickle.load(fileName)
# fileName.close()
#
# fileName = open('results/active_coarse','rb')
# ActiveBatches_coarse = pickle.load(fileName)
# fileName.close()
#
# x = []
# y = []
# count = 1
# for result in ActiveBatches_fine:
#     if(result[0] == count):
#         x.append(result[0])
#         y.append(result[2])
#         count+=1
# plt.plot(x,y, label = 'active/fine')
#
#
# x = []
# y = []
# count = 1
# for result in ActiveBatches_coarse:
#     if(result[0] == count):
#         x.append(result[0])
#         y.append(result[2])
#         count+=1
# plt.plot(x,y, label = 'active/coarse')
#
#
#
# plt.ylabel('PR-AUC')
# plt.xlabel('Iteration')
# plt.title('Active vs. Passive Learning')
# plt.legend(loc="lower right")
# plt.savefig('results/ActiveVsPassive.png')
#
