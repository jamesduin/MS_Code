import numpy as np
from matplotlib import pyplot as plt
import pickle

fileName = open('results/ActiveBatches20_fine','rb')
ActiveBatches20_fine = pickle.load(fileName)
fileName.close()

fileName = open('results/ActiveBatches20_coarse','rb')
ActiveBatches20_coarse = pickle.load(fileName)
fileName.close()

fileName = open('results/PassiveBatches20_fine','rb')
PassiveBatches20_fine = pickle.load(fileName)
fileName.close()

fileName = open('results/PassiveBatches20_coarse','rb')
PassiveBatches20_coarse = pickle.load(fileName)
fileName.close()

##### Print this folds pr_curve for fine
# precision = dict()
# recall = dict()
# average_precision = dict()
# precision["micro"], recall["micro"], _ = precision_recall_curve(y_testBin.ravel(),y_score.ravel())
# average_precision["micro"] = average_precision_score(y_testBin, y_score, average="micro")
#
# # Plot Precision-Recall curve
# plt.clf()
# plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
#          label='micro-average Precision-recall curve (area = {0:0.2f})'
#                ''.format(average_precision["micro"]))
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Precision-Recall')
# plt.legend(loc="lower left")
# plt.savefig(level+'_results/rnd'+str(rndNum)+'_'+level+'_PR.png')
# plt.clf()
# plt.close()


plt.figure()
#with plt.style.context('fivethirtyeight'):
plt.style.use('ggplot')
x = []
y = []
count = 1
for result in ActiveBatches20_coarse:
    if(result[0] == count):
        x.append(result[0])
        y.append(result[2])
        count+=1
plt.plot(x,y, label = 'active/coarse')

x = []
y = []
count = 1
for result in ActiveBatches20_fine:
    if(result[0] == count):
        x.append(result[0])
        y.append(result[2])
        count+=1
plt.plot(x,y, label = 'active/fine')

x = []
y = []
count = 1
for result in PassiveBatches20_coarse:
    if(result[0] == count):
        x.append(result[0])
        y.append(result[2])
        count+=1
plt.plot(x,y, label = 'passive/coarse')

x = []
y = []
count = 1
for result in PassiveBatches20_fine:
    if(result[0] == count):
        x.append(result[0])
        y.append(result[2])
        count+=1
plt.plot(x,y, label = 'passive/fine')


plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('Active vs. Passive Learning')
plt.legend(loc="lower right")
plt.savefig('results/ActiveVsPassive.png')

#
#
# f = open('results/_rnds.txt', 'w')
# results = []
# f.write('{:^20}       {:^20}\n'.format('coarse','fine'))
# for i,result in enumerate(coarse_rnds):
#     if(type(result[0]) != str and coarse_rnds[i][0] == fine_rnds[i][0]):
#         f.write('{0:<4}{1:<5.3f},{2:<5.3f},{3:<5.3f}        {4:<4}{5:<5.3f},{6:<5.3f},{7:<5.3f}\n'.format(coarse_rnds[
#                                                                                                               i][0],
#                                                                                                           coarse_rnds[
#                                                                                                               i][1],
#                                                                                                           coarse_rnds[
#                                                                                                               i][2],
#                                                                                                           coarse_rnds[
#                                                                                                               i][3],
#                                                                                                           fine_rnds[i][
#                                                                                                               0],
#                                                                                                           fine_rnds[i][
#                                                                                                               1],
#                                                                                                           fine_rnds[i][
#                                                                                                               2],
#                                                                                                           fine_rnds[i][
#                                                                                                               3]))
# f.close()
#
# ###### Save coarse results to a file
# f = open('results/_coarseResults.txt', 'w')
# for result in coarse_rnds:
#     # if (type(result[2]) == str):
#     #     f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
#     # elif (len(result) == 5):
#     #     f.write('{0:<5}{1:<10.3f}{2:<10.3f}{3:<10.3f} \n'.format(*result))
#     #else:
#     f.write(str(result) + '\n')
# f.close()
#
# ###### Save results to a file
# f = open('results/_fineResults.txt', 'w')
# for result in fine_rnds:
#     # if (type(result[2]) == str):
#     #     f.write('{0:5}{1:5}{2:10}{3:10} \n'.format(*result))
#     # elif (len(result) == 5):
#     #     f.write('{0:<5}{1:<10.3f}{2:<10.3f}{3:<10.3f} \n'.format(*result))
#     #else:
#     f.write(str(result)+'\n')
# f.close()