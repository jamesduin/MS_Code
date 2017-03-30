import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob
import re
import os
import matplotlib.colors as colors
import matplotlib.cm as cmx
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'Users' or rootDir == 'py'):
    dataDir = '../'
else:
    os.chdir('/work/scott/jamesd/')
    dataDir = '/home/scott/jamesd/MS_Code/'

rndSel = 120
#rndSel = 100
#rndSel = 45
#rndSel = 40

costs = {1.0,1.1,1.2,
        1.5,2.0,4.0,
        8.0,16.0,32.0,64.0}
#costs = {1.0,1.1}
ffrLabels = ['FFR[0.0]','FFR[0.1]','FFR[0.2]','FFR[0.3]','FFR[0.4]',
             'FFR[0.5]','FFR[0.6]','FFR[0.7]','FFR[0.8]','FFR[0.9]','FFR[1.0]','BANDIT']



def main():

    banditRes = np.array(pickle.load(open('BanditRndSel_'+str(rndSel)+'.res', 'rb')))
    plt.figure()
    plt.style.use('ggplot')
    for linInd,i in enumerate(range(0,48,4)):
        x = np.array(banditRes[:,i+2])
        x = np.log(x.astype(float))
        y = np.array(banditRes[:,i+3])
        # cVal = tuple(np.multiply(cVals[linInd],(1/270,1/270,1/270,1.0)))
        cVal = cVals[linInd]
        plt.plot(x, y, label=ffrLabels[linInd], linewidth=1.8,
                 fillstyle='none',
                 color=cVal, dashes=lineSty[linInd],
                 # marker=markSty[linInd],
                 # markersize=6,
                 # markeredgecolor=cVal,
                 # markeredgewidth=1.0,
                 # markevery=markEvSty[linInd]
                 )
        #leg = plt.legend(fancybox=True)

        axes = plt.gca()

        axes.set_xlim([0.0,4.5])
        # axes.set_ylim([0.842, 0.875])
        # axes.set_ylim(Ylims)


    plt.ylabel('PR-AUC')
    plt.xlabel('Log fine cost')
    title = 'Bandit Cost Analysis'
    plt.title(title)
    #plt.legend(loc="lower right")
    leg = plt.legend(bbox_to_anchor=(0.8, 0.675), loc=2, borderaxespad=0.)
    # set the alpha value of the legend: it will be translucent
    leg.get_frame().set_alpha(0.0)
    plt.savefig('BanditPlotLogFine.png')
    plt.close()


    #print(banditRes)
    f = open('output.txt', 'w')
    #printResultMat(f, resultMat)
    for rec in banditRes:
        for item in rec:
            f.write('{}'.format(item)+',')
        #f.write(str(rec)+'\n')
        f.write('\n')

    resultMat = []
    for linInd, i in enumerate(range(0,10)):
        print(i)
        a = banditRes[i].reshape(12,4)
        #print(a)
        a = a[a[:, 3].argsort()[::-1]]
        d_ind = np.array(range(0,len(a))).reshape(len(a), 1)
        a = np.hstack((d_ind, a))
        #print(a)
        a = a[a[:, 1].argsort()]
        print(a)
        b = a[:,1].reshape(1,12)
        c = a[:,0].reshape(1,12)
        print(b)
        print(c)
        if len(resultMat) == 0:
            resultMat = c
        else:
            resultMat = np.vstack((resultMat,c))

    resultMat = resultMat.astype(float)
    print(resultMat.shape)
    print(resultMat)

    min = (np.min(resultMat, axis=0))
    max = (np.max(resultMat, axis=0))
    mean = (np.mean(resultMat,axis=0))
    std = (np.std(resultMat, axis=0))
    tableMat = b.reshape((12,1))
    tableMat = np.hstack((tableMat, min.reshape(12, 1)))
    tableMat = np.hstack((tableMat,max.reshape(12,1)))
    tableMat = np.hstack((tableMat, mean.reshape(12, 1)))
    tableMat = np.hstack((tableMat, std.reshape(12, 1)))


    tableMat1 = tableMat[:]


    resultMatPR = []
    for linInd, i in enumerate(range(0, 10)):
        # print(i)
        a = banditRes[i].reshape(12, 4)
        #print(a)
        #a = a[a[:, 3].argsort()[::-1]]
        #d_ind = np.array(range(1, len(a) + 1)).reshape(len(a), 1)
        #a = np.hstack((d_ind, a))
        print(a)
        maxNum = np.max(a[:,3].astype(float))
        print(maxNum)
        #a = a[a[:, 1].argsort()]
        # print(a)
        b = a[:, 0].reshape(1, 12)
        c = np.abs(a[:, 3].reshape(1, 12).astype(float) - maxNum)
        print(c)
        if len(resultMatPR) == 0:
            resultMatPR = c
        else:
            resultMatPR = np.vstack((resultMatPR, c))
    print(resultMatPR)
    resultMatPR = resultMatPR.astype(float)
    tableMat = b.reshape((12, 1))
    min = (np.min(resultMatPR, axis=0))
    max = (np.max(resultMatPR, axis=0))
    mean = (np.mean(resultMatPR,axis=0))
    std = (np.std(resultMatPR, axis=0))
    tableMat = np.hstack((tableMat, min.reshape(12, 1)))
    tableMat = np.hstack((tableMat, max.reshape(12, 1)))
    tableMat = np.hstack((tableMat, mean.reshape(12, 1)))
    tableMat = np.hstack((tableMat, std.reshape(12, 1)))
    tableMat2 = tableMat[:]
    print(tableMat1)
    print(tableMat2)
    tableMat = np.hstack((tableMat2,tableMat1[:,1:]))
    #tableMat = np.hstack((b.reshape((12, 1)), tableMat))
    printResultMat(f, tableMat)
    print(tableMat)
    f.close()







cNorm  = colors.Normalize(vmin=-0.0, vmax=1.1)
# plt.style.use('ggplot')
# cmap = plt.get_cmap('rainbow')
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
lineSty = [[8,1],[4,1],[2,1],
           [8, 1], [4, 1], [2, 1],
           [8, 1], [4, 1], [2, 1],[4, 1],
           [8, 1],[8, 1]]
markSty = ['s','8','>',
           's','8','>',
           's','8','>','8',
           's','s']
markEvSty = [(1,8),(2,8),(3,8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (3, 8),
             (1, 8), (2, 8), (1, 8)]

cVals = [(0.90000000000000002, 0.25162433333706963, 0.12708553664078234, 1.0),
(0.90000000000000002, 0.48317993787976221, 0.25162433333706957, 1.0),
(0.86294117647058821, 0.67619870060118625, 0.3711206857265133, 1.0),
(0.69352941176470584, 0.81992038432147962, 0.48784802082520451, 1.0),
(0.53117647058823525, 0.89098219195263628, 0.58975546501210829, 1.0),
(0.36882352941176472, 0.89098219195263628, 0.67984446124709452, 1.0),
(0.2064705882352941, 0.8199203843214794, 0.75630966447410342, 1.0),
(0.037058823529411734, 0.67619870060118592, 0.81992038432147951, 1.0),
(0.12529411764705883, 0.48317993787976199, 0.86410948083716543, 1.0),
(0.28764705882352942, 0.25162433333706941, 0.89098219195263628, 1.0),
(0.45000000000000001, 0.0, 0.90000000000000002, 1.0),
 (0.0, 0.0, 0.0, 1.0)]
# cVals = [
# (255.*.9,255*.25, 255*.12, 1.0),
# (254., 50., 7., 1.0),
# (254., 102., 0., 1.0),
# (254., 169., 0., 1.0),
# (255., 209., 32., 1.0),
# (10., 207., 0., 1.0),
# (7., 142., 23., 1.0),
# (11., 156., 152., 1.0),
# (10., 0., 214., 1.0),
# (67., 0., 104., 1.0),
# (0., 0., 0., 1.0)
# ]


def printResultMat(f,resultMat):
    f.write('\FloatBarrier\n')
    f.write('\\begin{table}[H]\n')
    f.write('\centering\n')
    f.write('\caption{BanditCostAnalysis}\n')
    f.write('\label{tab:BanditCostAnalysis}\n')
    f.write('\\begin{tabular}{')
    for i in range(len(resultMat[0])):
        f.write('|l|')
    f.write('}\\toprule\n')

    for i, row in enumerate(range(len(resultMat))):
        for j, col in enumerate(range(len(resultMat[0])-1)):
            if(j == 5 or j == 6):
                f.write('{:.0f} & '.format(resultMat[i][j].astype(float)))
            elif(j==0):
                f.write('FFR[{}] & '.format(resultMat[i][j]))
            else:
                f.write('{:.3f} & '.format(resultMat[i][j].astype(float)))
        if (j == 5 or j == 6):
            f.write('{:.0f} \\\\'.format(resultMat[i][-1].astype(float)))
        elif(j==0):
            f.write('FFR[{}] \\\\'.format(resultMat[i][-1]))
        else:
            f.write('{:.3f} \\\\'.format(resultMat[i][-1].astype(float)))

        # if(i == 0):
        #     f.write('{} \\\\ \\midrule'.format(resultMat[-1][j]))
        # elif(i == len(resultMat[0]) -1 ):
        #     f.write('{} \\\\ \\bottomrule'.format(resultMat[-1][j]))
        # else:

        f.write('\n')
    f.write('\end{tabular}\n')
    f.write('\end{table}\n')
    f.write('\FloatBarrier\n')
    f.write('\n')


if __name__ == '__main__':
    main()

