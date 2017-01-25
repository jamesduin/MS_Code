import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,f_classif
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib



data = []
#partition_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
partition_list = [1, 2, 3]
#partition_list.remove(1)
for x in partition_list:
    partition = np.loadtxt("../data/partition_" + str(x))
    if data == []:
        data = partition
    else:
        data = np.vstack((partition, data))
y_train, X_trainPreScale = data[:, 0], data[:, 1:data.shape[1]]
# y_trainCoarse = []
# for i in y_train:
#     if i > 0:
#         y_trainCoarse.append(1.)
#     else:
#         y_trainCoarse.append(i)

#### scale dataset
scaler = preprocessing.StandardScaler().fit(X_trainPreScale)
X_trainFull = scaler.transform(X_trainPreScale)
selector = SelectPercentile(f_classif, percentile=75)
selector.fit(X_trainFull, y_train)
X_train = selector.transform(X_trainFull)



np.set_printoptions(precision=3)
clf = joblib.load('../HAL_standard/fine_models/fine_fold_1.pkl')
X_train_pred = clf.predict(X_train)
X_train_dec_fcn = clf.decision_function(X_train)
print('{0:10} {1:10}'.format('Predict','Dec Fcn'))
for i,pred in enumerate(X_train_pred):
    print('{0:10} {1:10}'.format(X_train_pred[i], str(X_train_dec_fcn[i])))
