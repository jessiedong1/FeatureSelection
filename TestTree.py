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
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\Lymph.csv')
ad.head()

#get all the features (I'm still looking for an efficient way)
X_data = ad.iloc[:, 0:27]
X = pd.DataFrame(X_data)

#Print the colums name
print("Features before feature selection: {}".format(X.columns.values))


y_data = ad['Label']
y = pd.DataFrame(y_data)
y=y.values.ravel()

"""
#Convert dataframe to numpy array
arr_ip = [tuple(i) for i in X.as_matrix()]
X = np.stack(arr_ip)
# dtyp = np.dtype(list(zip(X.dtypes.index, X.dtypes)))
# X = np.array(arr_ip, dtype=dtyp)


arr_ipy = [tuple(i) for i in y.as_matrix()]
y = [i[0] for i in arr_ipy]
"""

# dtypy = np.dtype(list(zip(y.dtypes.index, y.dtypes)))
# y = np.array(arr_ipy, dtype=dtypy)



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

#Get accurancy /auc
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set DT Before: {:.3f}",format(tree.score(X_train,y_train)))
print("Accuracy on test set DT Before: {:.3f}",format(tree.score(X_test,y_test)))
print("Selected feature in DT: {}".format())

#MLP CLASSIFIER
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on training set MLP Before: {:2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set MLP Before: {:.2f}".format(mlp.score(X_test, y_test)))

#Fisher score
score = fisher_score.fisher_score(X_train, y_train)
print(len(score))
idx = fisher_score.feature_ranking(score)
print(idx)
num_fea = 5

#Have to explain why the machine pick up those and do the classification again
#X1 = ad[['NEK6','SLC2A4','SLC2A5','SUV_C34', 'SUVreduction']]
#data.iloc[[0,3,6,24], [0,5,6]]
X1 = X.iloc[:, [idx[0], idx[1], idx[2], idx[3], idx[4]]]

X1 = pd.DataFrame(X1)
print("Selected features {}".format(X1.columns.values))
print("After: {}".format(X.iloc[0, [idx[0], idx[1], idx[2], idx[3],idx[4]]]))
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.3, random_state = 40)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set DT : {:.3f}",format(tree.score(X_train,y_train)))
print("Accuracy on test set DT: {:.3f}",format(tree.score(X_test,y_test)))


#MLP Classifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on training set MLP After: {:2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set MLP After: {:.2f}".format(mlp.score(X_test, y_test)))

#k nerighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions K Neighbors: {}".format(clf.score(X_test, y_test)))


#Visualize decision trees
"""
export_graphviz(tree, out_file="tree.dot", class_names = ["responese", "non-responese"],
                feature_names=['NEK6','SLC2A4','SLC2A5','SUV_C34', 'SUVreduction'], impurity=False, filled=True
                )

export_graphviz(tree, out_file="tree.dot", class_names = ["responese", "non-responese"] ,
                feature_names=X1.columns.values, impurity=False, filled=True
                )

with open("tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)
"""

