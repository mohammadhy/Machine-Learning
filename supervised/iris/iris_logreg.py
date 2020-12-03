"""
Created on Sun Nov  8 18:05:12 2020
@author: HY
python version 3.8.5
version tensorflow = 2.3.1

Explanation: 
In This Case we Have 2 Option 
First Use Regression Logistic But In This Case We Must Have 2 Class In Line 19 
Can See 
Second Use Softmax Regression In This Case Just Need Line 28 Multi_class And Without
Change On Target

"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
x = iris['data'][:, (2, 3)]     #petal width
y = (iris['target'] == 2).astype(np.int) #if verginiva 1 else 0(change it to 2 class)
#for softmax regression:
y0 = iris['target']

#task2: import linearregression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
log_reg.fit(x, y0)

'''
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_pre = log_reg.predict_proba(x_new)
plt.plot(x_new, y_pre[:, 1], 'g', label='verginica')
plt.plot(x_new, y_pre[:, 0], 'b--', label='no verginica')
'''
