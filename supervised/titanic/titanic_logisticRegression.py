#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:51:15 2020

@author: hassan
"""

import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt
'''
data = pd.read_csv('/home/hassan/python_code/pandas/StudentsPerformance.csv')


age = [random.randint(19, 87) for i in range(1000)]
age = pd.Series(age, name='age')

data = data.join(age)

bins=[18, 40, 90]

classifie_age = pd.cut(data['age'], bins, labels=['young', 'old'])

def g(tt):
    if tt in ['young']:
        return 0
    elif tt in ['old']:
        return 1
    
data['age'] = classifie_age.apply(g)
plt.pie(data.groupby('age').size(),explode=(0.1,0) ,autopct='%1.1f%%', shadow=True, labels=['young', 'old'])
'''
data = pd.read_csv('/home/hassan/Desktop/Machine_Learning_l/Files_and_codes/S03/code-3/train_Titanic.csv')

data['Name'] = data['Name'].apply(lambda x: x.split(',')[1].strip()[:3])

def g(tt):
    if tt in ['Mr.']:
        return 1
    elif tt in ['Miss']:
        return 2
    elif tt in ['Mrs']:
        return 3
    else:
        return 0

data['Name'] = data['Name'].apply(g)

data.drop('Cabin', inplace=True, axis=1)

data['Age'] = data['Age'].fillna(method='ffill')


data['Sex'] = pd.get_dummies(data['Sex'])
print(data.groupby('Sex').size())

embark = pd.get_dummies(data['Embarked'])

embark.drop(['S'], inplace=True, axis='columns')

data = data.join(embark)

data.drop(['Embarked','PassengerId', 'Ticket'], inplace=True, axis='columns')
data.pivot_table(index='Name', values='Sex', columns='Survived', aggfunc='size',fill_value=0)
print(data.pivot_table(index='Pclass', values='Sex', columns='Survived', aggfunc='size'))
plt.pie(data.groupby('Survived').size(),explode=(0.1,0) ,autopct='%1.1f%%', shadow=True, labels=['alive', 'dead'])


how_f_m = data.groupby(['Pclass', 'Sex'])['Survived'].count()
plt.figure()
plt.pie(data.groupby('Pclass').size(),explode=(0.0,0,0.1) ,autopct='%1.1f%%', shadow=True, labels=['1', '2', '3'])


from sklearn.linear_model import LogisticRegression

X_train = data.drop("Survived",axis=1)  
Y_train = data["Survived"]

model =  LogisticRegression()

model.fit(X_train, Y_train)

print(model.score(X_train, Y_train))

#80% score 













