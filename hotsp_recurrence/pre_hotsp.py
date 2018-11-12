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
from sklearn import model_selection
from sklearn import metrics

#import data from csv
ad = pd.read_csv('D:\Spring2017\Artificial Intelligence\hotspotlungtumtemp_Recurrence.csv')
ad.head()

#get all the features
X_data = ad.iloc[:, 0:52]
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

#np.save('Cervical_x', X_resampled)
#np.save('Cervical_y', y_resampled)

#Save the resmapling data into npy
X_resampled = np.load('Cervical_x.npy')
y_resampled = np.load('Cervical_y.npy')



#Put resmaple data to dataframe
#X_resampled = pd.DataFrame(X_resampled)
#X_resampled.columns = X.columns.values

#print(X_resampled)
print("The shape of X after applying ADASYN {}".format(X_resampled.shape))
print("The shape of y after applying ADASYN {}".format(y_resampled.shape))
print("The classes distribution {}".format(y_resampled))

#print(sorted(metrics.SCORERS.keys()))

classifier = DecisionTreeClassifier(random_state=0)

cv=StratifiedKFold(n_splits=10)

scores = cross_val_score(classifier, X_resampled, y_resampled, scoring="roc_auc", cv=cv)
acc = cross_val_score(classifier, X_resampled, y_resampled, scoring="accuracy", cv=cv)
recall = cross_val_score(classifier, X_resampled, y_resampled, scoring="recall", cv=cv)
prec = cross_val_score(classifier, X_resampled, y_resampled, scoring="precision", cv=cv)
#acc = cross_val_score(classifier, X_resampled, y_resampled, scoring="accuracy", cv=cv)
#print(scores)
print("Accuracy {}".format(acc.mean()))
print("AUC {}".format(scores.mean()))
print("Recall {}".format(recall.mean()))
print("Precision {}".format(prec.mean()))

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X_resampled, y_resampled):
    probas_ = classifier.fit(X_resampled[train], y_resampled[train]).predict_proba(X_resampled[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_resampled[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)
             #label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

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
plt.title('ROC in Cervical Before FS')
plt.legend(loc="lower right")
plt.show()
