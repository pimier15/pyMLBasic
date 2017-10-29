import numpy as np
from sklearn.tree import DecisionTreeClassifier
from Common.LoadIris import LoadIris
from Common.Plot import plot_decision_resion
from sklearn.tree import export_graphviz


tree = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 3 , random_state = 0)

X,y,XTrain,yTrain,XTest,yTest = LoadIris()

tree.fit(XTrain,yTrain)

plot_decision_resion(X,y,tree,test_idx=range(100,149))

export_graphviz(tree,out_file='tree.dot',feature_names=['petal length','petal width'])


