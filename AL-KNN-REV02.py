# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:21:09 2019

@author: Ali
"""

###############################################################################
from   sklearn.model_selection import train_test_split
from   sklearn.neighbors import KNeighborsClassifier
from   sklearn.metrics import classification_report, confusion_matrix
from   sklearn import preprocessing, metrics 
DATASET    = "D:\\Power_dataset.csv"
TARGETFILE = "D:\\TrainLabels.csv"
LABELFILE  = "D:\\Targets.csv"
LEN  = int(4392); # 
L1   = int(50*8.333)
L2   = int(105*8.333)
L3   = int(207*8.333)
L4   = int(320*8.333)
seed = 9
X        = np.loadtxt(DATASET, delimiter=",")
X_abs    = np.absolute(X)
X_array  = np.loadtxt(DATASET, delimiter=",")
X_abs    = np.absolute(X_array)
y_array  = np.loadtxt(TARGETFILE, delimiter=",")
X_red    = X_abs[:,L2:L3]
y_array  = y_array.astype('int')

#X_red   = np.concatenate((X_abs[:,0:L1],X_abs[:,L2:L3],X_abs[:,L4:LEN-1]),axis=1)
################################## Creating Model##########################################
X_train, X_test, y_train,y_test = train_test_split(X_abs,y_array, test_size=0.15, random_state=seed)
KNNclassifier = KNeighborsClassifier(n_neighbors=9, weights= 'uniform', algorithm='auto', leaf_size=243, metric= 'minkowski',n_jobs=-1 )
KNNclassifier.fit(X_train, y_train)
y_pred = KNNclassifier.predict(X_test)
print ("###########################Linear Model")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("F1 score is : %.3f" %metrics.f1_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))
print("precision is: %.3f " %metrics.precision_score(y_test,y_pred, average='weighted'))
print("Accuracy is : %.3f" %metrics.accuracy_score(y_test,y_pred))
print("##################################################")
'''
k_range = range(1,20)
scores = np.array(range(20))
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=9, weights= 'uniform', algorithm='auto', leaf_size=243, metric= 'minkowski',n_jobs=-1)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))   
plt.plot(k_range,scores_list)
plt.xlabel("Value of k for KNN")
plt.ylabel("Testing accuracy")
'''
     