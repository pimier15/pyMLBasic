from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def LoadIris():
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
    
    X = np.vstack((xTrainStd,xTestStd))
    Y = np.hstack((yTrain,yTest))

    return X,Y,xTrainStd,yTrain,xTestStd,yTest

