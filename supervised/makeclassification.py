#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:57:07 2020

@author: hassan

compare two model to each other with confuction matrix
and AUC score and ROC

"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.linear_model import SGDClassifier
print('#task1: creat a dataset')
from sklearn.datasets import make_classification
print('#task1.1: we have 20000 sample with 20 feature in 3 different class')
X, y = make_classification(n_samples=20000,n_classes=2  ,n_features=20, n_redundant=0, n_informative=2,
                         n_clusters_per_class=1)
print('#task1.2: split data into train and test with 70% train model and 30% test it')
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=1)
print('#task1.3: creat model knn')
model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainX, trainy)
P = model.predict_proba(testX)
print('#task2: calculate value score cross validation')
y_predict = cross_val_predict(model, trainX, trainy, cv=4)
print(cross_val_score(model, trainX, trainy, scoring='accuracy'))
print('#task3: auc score and fpr and tpr')

print('this for knn:')
print(precision_score(trainy, y_predict))
print(f1_score(trainy, y_predict))
print(recall_score(trainy, y_predict))
print(confusion_matrix(trainy, y_predict))

print('task4: create SGD model: ')
modelSGD = SGDClassifier(random_state=1)
modelSGD.fit(trainX, trainy)
pSGD = modelSGD.predict(testX)
y_predict_SGD = cross_val_predict(modelSGD, trainX, trainy, cv=4)
print(cross_val_score(modelSGD, trainX, trainy, scoring='accuracy'))
aucSGD = roc_auc_score(testy, pSGD)
fprSGD, tprSGD, thresholdsSGD = roc_curve(testy, pSGD)

print('task5: plot both model to see the better performance:')
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.plot(fprSGD, tprSGD, linestyle='solid', marker='*')
pyplot.grid()

print('''
      the best classifier is which model has just a TP and TN 
      in confuction matrix
      also in plot near the x,y = (1,0)
      and also see the precicion score and recall score 
      
      '''
      )

print('this for SGDClassifier:')

print(precision_score(trainy, y_predict_SGD))
print(f1_score(trainy, y_predict_SGD))
print(recall_score(trainy, y_predict_SGD))
print(confusion_matrix(trainy, y_predict_SGD))

# we see as a result the SGD it has better performance 



