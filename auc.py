print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# #############################################################################
# Data IO and generation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import LeaveOneOut
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz
#import data from csv
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\hotspotlungtumtemp_Recurrence.csv')
ad.head()

#get all the features (I'm still looking for an efficient way)
X_data = ad.iloc[:, 0:52]
X = pd.DataFrame(X_data)

#Print the colums name
print("Features before feature selection: {}".format(X.columns.values))


y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()

# Import some data to play with
"""
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
print(X)
print(y)
"""
ada = ADASYN()
X_resampled, y_resampled = ada.fit_sample(X, y)
np.save('cervical_x', X_resampled)
np.save('resample_y', y_resampled)
#Put resmaple data to dataframe
X_resampled = pd.DataFrame(X_resampled)
X_resampled.columns = X.columns.values
n_samples, n_features = X.shape
print(X_resampled.shape)
print(X.shape)
# Add noisy features
#random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, 20 * n_features)]
X = np.c_[X_resampled]
print(X.shape)
y=y_resampled
print(y.shape)
# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel='linear', probability=True,
                    # random_state=random_state)
#classifier = DecisionTreeClassifier(random_state=0)
classifier = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(classifier, X, y, scoring="accuracy", cv=cv)
print(scores)
print(scores.mean())

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC in Cervical Cancer Data Before FS')
plt.legend(loc="lower right")
plt.show()


#feature selection
for train, test in cv.split(X, y):
    score = fisher_score.fisher_score(X[train], y[train])
  #  probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve

print(len(score))
idx = fisher_score.feature_ranking(score)
#print(idx)
num_fea = 6

#Have to explain why the machine pick up those and do the classification again
#X1 = ad[['NEK6','SLC2A4','SLC2A5','SUV_C34', 'SUVreduction']]
#data.iloc[[0,3,6,24], [0,5,6]]
X = pd.DataFrame(X)
X1 = X.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], idx[10], idx[11]]]
print("After: {}".format(X_data.iloc[0, [idx[0], idx[1], idx[2], idx[3],idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], idx[10], idx[11]]]))

#X1 = X.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4]]]
X1 = np.c_[X1]
clf = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(clf, X1, y, scoring="accuracy", cv=cv)
print(scores)
print(scores.mean())
i = 0
for train, test in cv.split(X1, y):
    probas_ = clf.fit(X1[train], y[train]).predict_proba(X1[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC in Cervical Cancer Data After FS')
plt.legend(loc="lower right")
plt.show()
"""
# Relief
from skfeature.function.similarity_based import reliefF
for train, test in cv.split(X, y):
    score = reliefF.reliefF(X[train], y[train])
  #  probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve

print(len(score))
idx1 = reliefF.feature_ranking(score)


#Have to explain why the machine pick up those and do the classification again
#X1 = ad[['NEK6','SLC2A4','SLC2A5','SUV_C34', 'SUVreduction']]
#data.iloc[[0,3,6,24], [0,5,6]]
X = pd.DataFrame(X)
print(X.shape)
X2 = X.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], idx[10], idx[11]]]
print("After: {}".format(X_data.iloc[0, [idx[0], idx[1], idx[2], idx[3],idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], idx[10], idx[11]]]))

#X1 = X.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4]]]
X2 = np.c_[X2]
clf = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(clf, X2, y, scoring="accuracy", cv=cv)
print(scores)
print(scores.mean())
i = 0
for train, test in cv.split(X2, y):
    probas_ = clf.fit(X2[train], y[train]).predict_proba(X2[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC in Cervical Cancer Data After FS')
plt.legend(loc="lower right")
plt.show()
"""