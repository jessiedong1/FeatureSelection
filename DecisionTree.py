import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import ADASYN
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import StratifiedKFold
#import data from csv
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\RTEP3OesoAvant2_DiseaseFree.csv')
ad.head()

#get all the features
X_data = ad.iloc[:, 0:29]
X = pd.DataFrame(X_data)

#Print the colums name
print("Features before feature selection: {}".format(X.columns.values))

#Get classes
y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()


#ADASYN TO resampling( I need cite this source) and also explain the reasons
ada = ADASYN()
X_resampled, y_resampled = ada.fit_sample(X, y)

np.save('lymph_x', X_resampled)
np.save('lymph_y', y_resampled)
"""
#Save the resmapling data into npy
X_resampled = np.load('lymph_x.npy')

y_resampled = np.load('lymph_y.npy')



#Put resmaple data to dataframe
X_resampled = pd.DataFrame(X_resampled)
X_resampled.columns = X.columns.values
"""

#print(X_resampled)
print("The shape of X after applying ADASYN {}".format(X_resampled.shape))
print("The shape of y after applying ADASYN {}".format(y_resampled.shape))
#print("The classes distribution {}".format(y))



classifier = DecisionTreeClassifier(random_state=0)

cv=StratifiedKFold(n_splits=10)
scores = cross_val_score(classifier, X_resampled, y_resampled, scoring="recall", cv=cv)
#acc = cross_val_score(classifier, X_resampled, y_resampled, scoring="accuracy", cv=cv)
#print(scores)
#print("Accuracy {}".format(acc.mean()))
print("AUC {}".format(scores.mean()))



X = np.c_[X_resampled]
y=y_resampled


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
accs = []
i = 0
for train, test in cv.split(X, y):
    clf = classifier.fit(X[train], y[train])
    probas_ = clf.predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    acc = clf.score(X[test], y[test])
    accs.append(acc)
    i += 1

print(sum(aucs) / float(len(aucs)))
print(sum(accs) / float(len(accs)))

#Fisher score
score = fisher_score.fisher_score(X, y)
#print(len(score))
idx = fisher_score.feature_ranking(score)
#print(idx)
num_fea = 6
X_resampled = pd.DataFrame(X_resampled)
X1 = X_resampled.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], idx[10], idx[11]]]

#X1 = X.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4]]]
X1 = pd.DataFrame(X1)
#print("Selected features {}".format(X1.columns.values))

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y_resampled, test_size = 0.3, random_state = 40)

#tree = DecisionTreeClassifier(random_state=0)
#tree.fit(X_train1, y_train1)
#print("Accuracy on training set DT : {:.3f}",format(tree.score(X_train1,y_train1)))
#print("Accuracy on test set DT: {:.3f}",format(tree.score(X_test1,y_test1)))


#LEAVE ONE OUT
tree1 = DecisionTreeClassifier(random_state=0)

scores1 = cross_val_score(tree1, X1, y_resampled, scoring="roc_auc", cv=cv)
acc2 = scores = cross_val_score(tree1, X1, y_resampled, scoring="accuracy", cv=cv)
print(acc2.mean() )
print(scores1)
print(scores1.mean())



