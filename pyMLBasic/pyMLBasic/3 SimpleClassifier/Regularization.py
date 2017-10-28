from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
X = np.vstack((xTrainStd,xTestStd))
Y = np.hstack((yTrain,yTest))

## Regularization Test Start ##
weight , parmas , lambdas = [] ,[] , []
for c in np.arange(-5,5):
    lr = LogisticRegression(C = 10.0**c , random_state=0)
    lr.fit(xTrainStd,yTrain)

    predict = lr.predict(xTestStd)
    
    print()
    print("Current Lambda is {0}".format(1/10.0**c))
    print("Current C is {0}".format(10.0**c))
    print("Fail Sample : {0}".format( (yTest != predict).sum() ))
    print("Accuracy : {0:.2f}".format(accuracy_score(predict , yTest)))


    weight.append(lr.coef_[1])
    parmas.append(10.0**c)
    lambdas.append(1/10.0**c)

weight = np.array(weight)

plt.plot(parmas , weight[:,0] ,label = 'weight1' )
plt.plot(parmas , weight[:,1] , linestyle = '--',label = 'weight2')
plt.xlabel('c')
plt.xscale('log')
plt.show()



