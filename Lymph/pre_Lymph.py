import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
#import data from csv
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\Lymph.csv')
ad.head()

#get all the features
X_data = ad.iloc[:, 0:27]
X = pd.DataFrame(X_data)

#Print the colums name
print("Features before feature selection: {}".format(X.columns.values))

#Get classes
y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()


#ADASYN TO resampling( I need cite this source) and also explain the reasons

#ada = ADASYN()
#X_resampled, y_resampled = ada.fit_sample(X, y)

#np.save('lymph_x', X_resampled)
#np.save('lymph_y', y_resampled)

#Save the resmapling data into npy
X_resampled = np.load('Lymph_x.npy')
y_resampled = np.load('Lymph_y.npy')


#Put resmaple data to dataframe
#X_resampled = pd.DataFrame(X_resampled)
#X_resampled.columns = X.columns.values

#print(X_resampled)
#print("The shape of X after applying ADASYN {}".format(X_resampled.shape))
#print("The shape of y after applying ADASYN {}".format(y_resampled.shape))
#print("The classes distribution {}".format(y_resampled))

#print(sorted(metrics.SCORERS.keys()))
#Stratified KFold
cv=StratifiedKFold(n_splits=13)

dtc = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(dtc, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc = cross_val_score(dtc, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall = cross_val_score(dtc, X_resampled, y_resampled, scoring="recall", cv=cv)
prec = cross_val_score(dtc, X_resampled, y_resampled, scoring="precision", cv=cv)
fmDT = cross_val_score(dtc, X_resampled, y_resampled, scoring="f1", cv=cv)
#print(metrics.SCORERS.keys())
#Results from DT
print("Accuracy in DT{}".format(acc.mean()))
print("AUC in DT {}".format(scores.mean()))
print("Recall in DT {}".format(recall.mean()))
print("Precision in DT {}".format(prec.mean()))
print("F-Measure in DT {}".format(fmDT.mean()))


#KNN
print()
knn = KNeighborsClassifier(n_neighbors=3)
auc_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="recall", cv=cv)
prec_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="precision", cv=cv)
fm_knn = cross_val_score(knn, X_resampled, y_resampled, scoring="f1", cv=cv)
#Results from KNN
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