#python3.8.5
"""
Created on Sat Nov  7 10:22:05 2020

@author: HY
"""

from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#task1: load mnist dataset
mnist = fetch_openml('mnist_784', version=1)


x, y = mnist['data'],mnist['target']

#task2: choose one digit from mnist and show on matplotlib
som_digit = x[5]
som_digit = som_digit.reshape(28, 28)
plt.imshow(som_digit, cmap='binary')
plt.axis('off')
plt.show()
#task3: change y type to uint8 
y = y.astype(np.uint8)

#task4: creat xtest xtrain ytest and ytrain 70% train data with 30% test 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=1)
#task4.1: choose all 5 lable in y_train5 and y_test5
y_train5 = (ytrain == 5)
y_test5 = (ytest == 5)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(xtrain, y_train5)


from sklearn.model_selection import cross_val_score
print(cross_val_score(knn, xtrain, y_train5, cv=3, scoring='accuracy'))



















