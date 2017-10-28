from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
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


model = Perceptron( n_iter = 40  , eta0 = 0.1  ,random_state = 0 )
model.fit(xTrainStd , yTrain) # use SGD optimization and MSE loss

predict = model.predict(xTestStd)
print("Fail Sample : {0}".format( (yTest != predict).sum() ))
print("Accuracy : {0:.2f}".format(accuracy_score(predict , yTest)))


''' plot data '''
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_resion(X,Y,classifier , test_idx = None, resolution = 0.02):
    markers = ('s' , 'x' , 'o' , '^' , 'v')
    colors = ('red' , 'blue' ,'lightgreen' ,'gray' , 'cyan')
    cmap = ListedColormap( colors[:len(np.unique(Y))])

    #plot decision region
    x1min ,x1max = X[: , 0].min() - 1 , X[: , 0].max() +1
    x2min ,x2max = X[: , 1].min() - 1 , X[: , 1].max() +1

    xx1 , xx2 = np.meshgrid( np.arange(x1min,x1max,resolution) , 
                              np.arange(x2min,x2max,resolution) )

    # xx1.shape = len(xx1) x xx2
    # xx2.shape = len(xx1) x xx2


    # xx12.ravel().length =  len(xx1) * len(xx2)
    # np.array([xx1.ravel() , xx2.ravel()]).shape = 2 x xx12.ravel().length
    # np.array([xx1.ravel() , xx2.ravel()]).T.shape = xx12.ravel().length x 2
    Z =  classifier.predict(np.array([xx1.ravel() , xx2.ravel()]).T) 
    Z = Z.reshape(xx1.shape)
    # now Z xx1 , xx2 has same shape

    plt.contourf(xx1 , xx2 , Z , alpha = 0.4 , cmap = cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot all sample
    print(X.shape)
    print(Y.shape)
    XTest,YTest = X[test_idx,:] , Y[test_idx]
    for idx , cl in enumerate(np.unique(Y)):
        plt.scatter(x = X[ Y == cl , 0] , y = X[ Y == cl , 1] , alpha = 0.8 , c = cmap(idx) , marker = markers[idx] , label = cl)

    plt.show()



X = np.vstack((xTrainStd,xTestStd))
Y = np.hstack((yTrain,yTest))
plot_decision_resion(X,Y,model ,range(100,130) )





