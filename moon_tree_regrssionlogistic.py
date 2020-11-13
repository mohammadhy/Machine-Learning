#python3.8.5
"""
Created on Thu Nov 12 12:12:47 2020

@author: hassan
"""

print('----------------------Moon Example---------------------------')
'''
import module that we need  
'''
from sklearn.datasets import make_moons
from sklearn.model_selection  import  train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import numpy as np
#Train tree :
x, y = make_moons(n_samples=1000, noise=0.4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
tree_moon = DecisionTreeRegressor(max_depth=3)
tree_moon.fit(x_train, y_train)
y_predict = tree_moon.predict(x_test)


#Train logistic regression:
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
log_reg.fit(x_train, y_train)

log_predict  = log_reg.predict(x_test)

#print(cross_val_score(log_reg, x_train, y_train, scoring='accuracy'))

#confusion_matrix from regression:
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(log_reg, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 )
plt.figure()
#use graphiz to visialize the trained decision:
''' run on commandline ==> dot -Tpng moon_tree.dot -o moon_tree.png '''

export_graphviz( 

        tree_moon,
        out_file=('moon_tree.dot'),
        rounded=True,
        filled=True)

print(f'this is for logistic regression accuracy_score ==> {accuracy_score(y_test, log_predict)}')

print(f'this is for decison treee accuracy_score ==> {accuracy_score(y_test, y_predict.round())}')


from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x, y, clf = tree_moon , legend=2 )
plt.figure()
plot_decision_regions(x, y, clf = log_reg, legend=2 )


'''
from mpl_toolkits.mplot3d import Axes3D
X ,Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2
fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
s = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=plt.cm.coolwarm)
fig.colorbar(s, shrink=0.5, aspect=5)
'''
'''
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[0, 4, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="class_0")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="class_1")
        plt.axis(axes)
    if iris:
        plt.xlabel("featuare 0", fontsize=14)
        plt.ylabel("featuare 1", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_moon, x_train, y_train, legend=False)
plt.plot([0, 3.5], [0.8, 0.8], "k:", linewidth=2)
plt.plot([0, 3.5], [2, 2], "k--", linewidth=2)
plt.text(1.0, 0.9, "Depth=0", fontsize=15)
plt.text(1.0, 2.40, "Depth=1", fontsize=13)

plt.show()
'''
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_predict)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
print(confusion_matrix(y_test, y_predict.round()))
