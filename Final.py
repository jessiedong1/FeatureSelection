import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import LeaveOneOut
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#import data from csv
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\RTEP3OesoAvant2_DiseaseFree.csv')
ad.head()

#get all the features (I'm still looking for an efficient way)
X_data = ad.iloc[:, 0:29]
X = pd.DataFrame(X_data)
#Print the colums name
print("Features before feature selection: {}".format(X.columns.values))

#Print the colums name
y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()


#ADASYN TO resampling( I need cite this source) and also explain the reasons
pca = PCA(n_components=2)
ada = ADASYN()
X_resampled, y_resampled = ada.fit_sample(X, y)

#Put resmaple data to dataframe
X_resampled = pd.DataFrame(X_resampled)
X_resampled.columns = X.columns.values
#y_resampled = pd.DataFrame(y_resampled)
#y_resampled.columns = y.columns.values

#print(X_resampled)
print("The shape of X after applying ADASYN {}".format(X_resampled.shape))
print("The shape of y after applying ADASYN {}".format(y_resampled.shape))
#print("The classes distribution {}".format(y))

# Split the data into training data and testing data
#X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 40)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 40)

#tree1 = DecisionTreeClassifier(random_state=0)
#tree1.fit(X_train, y_train)
#print("Accuracy on training set DT Before: {:.3f}",format(tree1.score(X_train,y_train)))
#print("Accuracy on test set DT Before: {:.3f}",format(tree1.score(X_test,y_test)))

tree1 = DecisionTreeClassifier(random_state=0)
loo = KFold(n_splits=44)
scores = cross_val_score(tree1, X_resampled, y_resampled, cv=loo)
#print(scores)
print(scores.mean())

#Fisher score
score = fisher_score.fisher_score(X_train, y_train)
#print(len(score))
idx = fisher_score.feature_ranking(score)
#print(idx)
num_fea = 6

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
loo = KFold(n_splits=10)
scores = cross_val_score(tree1, X1, y_resampled, scoring="accuracy", cv=loo)
print(scores)
print(scores.mean())



