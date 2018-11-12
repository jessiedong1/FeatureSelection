import pandas as pd
from skfeature.function.information_theoretical_based import FCBF
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


#import data from csv
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\RTEP3OesoAvant2_DiseaseFree.csv')
ad.head()


#get all the features
X_data = ad.iloc[:, 0:29]
X = pd.DataFrame(X_data)

#Print the colums name
#print("Features before feature selection: {}".format(X.columns.values))

#Get classes
y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()

#Save the resmapling data into npy
X_resampled = np.load('free_x1.npy')
y_resampled = np.load('free_y1.npy')

X_resampled = pd.DataFrame(X_resampled)
X_resampled.columns = X.columns.values
X_resampled['Sex=M'] = X_resampled['Sex=M'].astype('bool')
X_resampled['OMS'] = X_resampled['OMS'].astype('bool')
X_resampled['Stade=ND'] = X_resampled['Stade=ND'].astype('bool')
X_resampled['Stade={2,2a,2b}'] = X_resampled['Stade={2,2a,2b}'].astype('bool')
X_resampled['Stade=3'] = X_resampled['Stade=3'].astype('bool')
X_resampled['1/3 sup'] = X_resampled['1/3 sup'].astype('bool')
X_resampled['1/3 moyen'] = X_resampled['1/3 moyen'].astype('bool')
X_resampled['1/3 inf'] = X_resampled['1/3 inf'].astype('bool')
X_resampled['Dysphagie=0'] = X_resampled['Dysphagie=0'].astype('bool')
X_resampled['Dysphagie=1'] = X_resampled['Dysphagie=1'].astype('bool')
X_resampled['Dysphagie=2'] = X_resampled['Dysphagie=2'].astype('bool')
X_resampled['Dysphagie=3'] = X_resampled['Dysphagie=3'].astype('bool')


# split data into 10 folds
cv=StratifiedKFold(n_splits=8)

# perform evaluation on classification task
num_fea = 11    # number of selected features
"""
for train, test in cv.split(X_resampled,y_resampled):
    # obtain the feature weight matrix
    idx = FCBF.fcbf(X_resampled.values[train], y_resampled[train], n_selected_features=num_fea)
    # sort the feature scores in an ascending order according to the feature scores
    print(idx)
"""
X_resampled = X_resampled.iloc[:, [0,8,13]]

print("Selected Features {}".format(X_resampled.columns.values))
#Decision Tree
dtc = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(dtc, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc = cross_val_score(dtc, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall = cross_val_score(dtc, X_resampled, y_resampled, scoring="recall", cv=cv)
prec = cross_val_score(dtc, X_resampled, y_resampled, scoring="precision", cv=cv)
fmDT = cross_val_score(dtc, X_resampled, y_resampled, scoring="f1", cv=cv)

#print(metrics.SCORERS.keys())
#Results from DT
print()
print("Accuracy in DT {}".format(acc.mean()))
print("AUC in DT {}".format(scores.mean()))
print("Recall in DT {}".format(recall.mean()))
print("Precision in DT {}".format(prec.mean()))
print("F-Measure in DT {}".format(fmDT.mean()))


#KNN
knn = KNeighborsClassifier(n_neighbors=3)
auc_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="recall", cv=cv)
prec_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="precision", cv=cv)
fm_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="f1", cv=cv)
#Results from KNN
print()
print("Accuracy in KNN {}".format(acc_knn.mean()))
print("AUC in KNN {}".format(auc_knn.mean()))
print("Recall in KNN {}".format(recall_knn.mean()))
print("Precision in KNN {}".format(prec_knn.mean()))
print("F-Measure in KNN {}".format(fm_knn.mean()))

#Linear SVC
print()
lsv = LinearSVC()
auc_lsv = cross_val_score(lsv, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc_lsv = cross_val_score(lsv, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall_lsv = cross_val_score(lsv, X_resampled, y_resampled, scoring="recall", cv=cv)
prec_lsv = cross_val_score(lsv, X_resampled, y_resampled, scoring="precision", cv=cv)
fm_lsv = cross_val_score(lsv, X_resampled, y_resampled, scoring="f1", cv=cv)
#Results from Linear SVC
print("Accuracy in LinearSVC {}".format(acc_lsv.mean()))
print("AUC in LinearSVC {}".format(auc_lsv.mean()))
print("Recall in LinearSVC {}".format(recall_lsv.mean()))
print("Precision in LinearSVC {}".format(prec_lsv.mean()))
print("F-Measure in LinearSVC {}".format(fm_lsv.mean()))

#Logistic Regression
print()
lr = LogisticRegression()
auc_lr = cross_val_score(lr, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc_lr = cross_val_score(lr, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall_lr = cross_val_score(lr, X_resampled, y_resampled, scoring="recall", cv=cv)
prec_lr = cross_val_score(lr, X_resampled, y_resampled, scoring="precision", cv=cv)
fm_lr = cross_val_score(lr, X_resampled, y_resampled, scoring="f1", cv=cv)
#Results from Logistic Regression
print("Accuracy in LogisticRegression {}".format(acc_lr.mean()))
print("AUC in LogisticRegression {}".format(auc_lr.mean()))
print("Recall in LogisticRegression {}".format(recall_lr.mean()))
print("Precision in LogisticRegression {}".format(prec_lr.mean()))
print("F-Measure in LogisticRegression {}".format(fm_lr.mean()))

#MLPC
print()
mlpc = MLPClassifier()
auc_mlpc = cross_val_score(mlpc, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc_mlpc = cross_val_score(mlpc, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall_mlpc = cross_val_score(mlpc, X_resampled, y_resampled, scoring="recall", cv=cv)
prec_mlpc = cross_val_score(mlpc, X_resampled, y_resampled, scoring="precision", cv=cv)
fm_mlpc = cross_val_score(mlpc, X_resampled, y_resampled, scoring="f1", cv=cv)
#Results from MLPC
print("Accuracy in MLPC {}".format(acc_mlpc.mean()))
print("AUC in MLPC {}".format(auc_mlpc.mean()))
print("Recall in MLPC {}".format(recall_mlpc.mean()))
print("Precision in MLPC {}".format(prec_mlpc.mean()))
print("F-Measure in MLPC {}".format(fm_mlpc.mean()))
