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


rndSel = 100

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
    f.write('\\begin{table}[h]\n')
    f.write('\centering\n')
    f.write('\\begin{tabular}{')
    for i in range(len(resultMat)):
        f.write('|l|')
    f.write('}\\toprule\n')
    for i, row in enumerate(resultMat[0]):
        for j, col in enumerate(resultMat[:-1]):
            if (isinstance(resultMat[j][i], str)):
                f.write('{} & '.format(resultMat[j][i]))
        if(i == 0):
            f.write('{} \\\\ \\midrule'.format(resultMat[-1][i]))
        elif(i == len(resultMat[0]) -1 ):
            f.write('{} \\\\ \\bottomrule'.format(resultMat[-1][i]))
        else:
            f.write('{} \\\\'.format(resultMat[-1][i]))
        f.write('\n')
    f.write('\end{tabular}\n')
    f.write('\caption{BanditCostAnalysis}\n')
    f.write('\label{tab:BanditCostAnalysis}\n')
    f.write('\end{table}\n')
    f.write('\FloatBarrier\n')
    f.write('\n')


if __name__ == '__main__':
    main()

