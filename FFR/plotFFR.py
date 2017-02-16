import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'Users' or rootDir == 'py'):
    dataDir = '../'
else:
    os.chdir('/work/scott/jamesd/')
    dataDir = '/home/scott/jamesd/MS_Code/'




resultsDir = '_results/results'
#resultsDir = 'resultsRBF11sclBy1_15/results'
#resultsDir = 'resultsSclBy1/results'
#resultsDir = 'results'




rndType = dict()
for fname in glob.glob(resultsDir+'/*.res'):
    type = re.split("[/\.]", fname)[-2]
    rndType[type] = []
    try:
        results = pickle.load(open(fname, 'rb'))
    except EOFError:
        pass
    for result in results:
        rndType[type].append(result)

max = []
for type in rndType:
    len(rndType[type])
#print(np.max(max))

for type in rndType:
    ### print out the folds
    f = open(resultsDir+'/_' + type + '.txt', 'w')
    for rec in rndType[type]:
        f.write(str(rec)+'\n')
    f.close()

#
# plt.figure()
# #with plt.style.context('fivethirtyeight'):
# plt.style.use('ggplot')
# for type in rndTypeSet:
#     prs = []
#     for fold in sorted(rndTypeFoldMat[type]):
#         prFold = []
#         for rec in rndTypeFoldMat[type][fold]:
#             #if(not isinstance(rec,str)):
#             if('pr'in rec):
#                 ind =[i for i in range(len(rec)) if rec[i] == 'pr']
#                 prFold.append(rec[ind[0]+1])
#         prFold = np.array(prFold).reshape(len(prFold),1)
#         if prs == []:
#             prs =prFold
#         else:
#             prs = np.hstack((prs,prFold))
#     x_pr = np.array(range(1,len(prs)+1))
#     y_pr = np.mean(prs, axis=1)
#     plt.plot(x_pr,y_pr, label = 'avg_'+type)
#
# plt.ylabel('PR-AUC')
# plt.xlabel('Iteration')
# plt.title('Active vs. Passive Learning')
# plt.legend(loc="lower right")
# plt.savefig(resultsDir+'/ActiveVsPassivePR.png')
#
#
