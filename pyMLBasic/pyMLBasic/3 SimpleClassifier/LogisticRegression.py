from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Common import Plot as pltD

# sklaern bunch type
iris = datasets.load_iris()

#ndarray type
X = iris.data[:,[2,3]]
Y = iris.target

xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size = 0.3 , random_state = 0 )

sc = StandardScaler()
sc.fit(xTrain) # calc labmda(mean) and sigma(std) along each feature

xTrainStd = sc.transform(xTrain)
xTestStd = sc.transform(xTest)

############ Data Load Complete ##########

lr = LogisticRegression( C = 1000.0 , random_state=0)
lr.fit(xTrainStd,yTrain)
predict = lr.predict(xTestStd)

print("Fail Sample : {0}".format( (yTest != predict).sum() ))
print("Accuracy : {0:.2f}".format(accuracy_score(predict , yTest)))

X = np.vstack((xTrainStd,xTestStd))
Y = np.hstack((yTrain,yTest))

pltD.plot_decision_resion(X , Y , lr , range(100,140) )


