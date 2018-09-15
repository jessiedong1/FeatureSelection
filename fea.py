import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\Lymph.csv')

ad.head()

X = ad[['MME', 'LRMP','NEK6', 'BCL6', 'LMO2', 'ITPKB', 'MYBL1', 'SLC2A1','SLC2A2', 'SLC2A3', 'SLC2A4', 'SLC2A5', 'SUV_Base', 'SUV_C34', 'Gly_C34', 'SUVreduction']]
y = ad['Label']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

from skfeature.function.similarity_based import fisher_score

score = fisher_score.fisher_score(X_train, y_train)

print(score)

idx = fisher_score.feature_ranking(score)

print(idx)

num_fea = 5

# train_features = np.reshape([features[i, :] for i in train], (len(train), 99))

selected_features_train = [X_train[:, i:i+1] for i in idx]
np.reshape(selected_features_train)

# selected_features_test = X_test[:, idx[0:num_fea]]

# print(selected_features_train)
# print(selected_features_test)

# from sklearn import svm

# clf = svm.LinearSVC()

# clf.fit(selected_features_train, y_train)

