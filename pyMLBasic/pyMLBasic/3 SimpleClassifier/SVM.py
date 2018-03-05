import numpy as np
from LoadIris import LoadIris
from Plot import plot_decision_resion

from sklearn.svm import SVC


X,Y,Xtrain,YTrain,XTest,YTest = LoadIris()
svm = SVC(kernel = 'linear' , C = 1.0 , random_state = 0)
svm.fit(Xtrain,YTrain)


plot_decision_resion(X,Y,svm , range(100,149))


