import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100
def predict(X, W, b):
    z = np.dot(X, W) + b
    p = 1 / (1 + np.exp(-z))
    return (p >= 0.5).astype(int)
def initialise(nf):
    W = np.zeros(nf)
    b = 0
    return W, b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
def logistic_regression(xi,W,b,yi):
    z = np.dot(xi, W) + b
    p = sigmoid(z)
    error = p - yi
    return error
def StochasticGradient(error, xi, W, b, lr):
    grad_W = error * xi
    grad_b = error
    W = W - lr * grad_W
    b = b - lr * grad_b
    return W, b

def model_V1(X, y, nf, epochs=500, lr=0.01):

    W, b = initialise(nf)
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            error = logistic_regression(xi,W,b,yi)
            W, b = StochasticGradient(error, xi, W, b, lr)
    return W, b
            
    