import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')

data = np.array(data)
# Rows (m) and features+1 (n)
m,n = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T # 1000 random samples for development, and transpose
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T # Rest for training, and transpose
Y_train = data_train[0]
X_train = data_train[1:n]

def initParameters():
    W1 = np.random.rand(10, 784)
    b1 = np.random.randn(10,1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.randn(10,1)
    return W1, b1, W2, b2

def forwardPropagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = np.maximum(Z1, 0) # ReLU
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def backwardPropagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[0]
    dZ2 = A2 - Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2
    return W1, b1, W2, b2

def gradientDescent(X, Y, learningRate, iterations):
    W1, b1, W2, b2 = initParameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardPropagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", accuracy(A2, Y))
    return W1, b1, W2, b2

def accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def predict(W1, b1, W2, b2, X):
    Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
    return np.argmax(A2, axis=0)

def plotCosts(costs, learningRate, iterations):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learningRate))
    plt.show()

def plotPredictions(X, Y, W1, b1, W2, b2):
    n = X.shape[1]
    predictions = predict(W1, b1, W2, b2, X)
    for i in range(n):
        plt.imshow(X[:,i].reshape(28,28), cmap='gray')
        plt.title("Prediction: " + str(predictions[i]) + ", Label: " + str(np.argmax(Y[:,i])))
        plt.show()

W1, b1, W2, b2 = gradientDescent(X_train, Y_train, 0.1, 1000)

