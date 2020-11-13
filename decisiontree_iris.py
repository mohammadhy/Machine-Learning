#python3.8.5
"""
Created on Thu Nov 12 10:26:46 2020

@author: HY
Make Decision Tree iris Dataset
"""
#task1: import module that we need:
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#task2: load dataset iris: 
iris = load_iris()

x = iris.data[:, 2:]
y = iris.target

'''
-decision tree dose not care about scaling 

-as you can see below to avoid overfitting the traing data we use max_depth and reduce the risk
of overfiting 

-also you can change 'gini' to entropy by pass criterion 

-other good hyperparameters presort=True speed up traning only if the datasets smaler than
a few thousand instances
'''
#task3: make classification model:
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(x, y)
#task4: use graphiz to visialize the trained decision:
''' run on commandline ==> dot -Tpng iris_tree.dot -o iris_tree.png '''

export_graphviz(

        tree,
        out_file=('iris_tree.dot'),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True)
#task4.1: another plot you can use:
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x, y, clf = tree , legend=2 )
#task5: predict model:
print(tree.predict_proba([[5, 1.5]]))

#task6: althouh you can use decision tree for regression task instead of predicting a class predict a value

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(x, y)

#task4: use graphiz to visialize the trained decision:
''' run on commandline ==> dot -Tpng iris_tree_reg.dot -o iris_tree_reg.png '''

export_graphviz( 

        tree_reg,
        out_file=('iris_tree_reg.dot'),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True)

