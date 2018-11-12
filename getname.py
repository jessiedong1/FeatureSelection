from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import fisher_score
import graphviz
from sklearn.tree import export_graphviz
#import data from csv
from sklearn.ensemble import GradientBoostingClassifier

ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\hotspotlungtumtemp_Recurrence.csv')
ad.head()

#get all the features (I'm still looking for an efficient way)
X_data = ad.iloc[:, 0:52]
X = pd.DataFrame(X_data)

#Print the colums name
#print("Features before feature selection: {}".format(X.columns.values))


y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
tree1 = DecisionTreeClassifier(random_state=0)
loo = KFold(n_splits=24)
scores = cross_val_score(tree1, X, y, cv=loo)

print(scores)
print(scores.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)

gbrt = GradientBoostingClassifier(random_state=40, learning_rate=0.01)

gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))