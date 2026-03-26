import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100
def predict(X, W, b, threshold=0.5):
    z = np.dot(X, W) + b
    p = 1 / (1 + np.exp(-z))
    return (p >= threshold).astype(int)
def threshold_search(X_test, y_test, W, b):
    
    z = np.dot(X_test, W) + b
    p = sigmoid(z)

    best_acc = 0
    best_t = 0.5

    for t in np.arange(0.30, 0.70, 0.01):
        y_pred = (p >= t).astype(int)
        acc = accuracy(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_t = t

    print("Best Threshold:", best_t)
    print("Best Accuracy:", best_acc)
    return best_t


def initialise(nf):
    W = np.zeros(nf)
    b = 0
    return W, b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
def logistic_regression(X,W,b,y):
    z = np.dot(X, W) + b
    p = sigmoid(z)
    error = p - y
    return error,p
def StochasticGradient(error, xi, W, b, lr):
    grad_W = error * xi
    grad_b = error
    W = W - lr * grad_W
    b = b - lr * grad_b
    return W, b
def BatchGradientDescent(error, X, W, b, lr):
    grad_W = np.dot(X.T, error) / len(X)
    grad_b = np.mean(error)
    W = W - lr * grad_W
    b = b - lr * grad_b
    return W,b
def BatchGradientDescentV2(error, X, W, b, lr,n_samples,lambda_=0.01):
    grad_W = (np.dot(X.T, error) / n_samples) + lambda_ * W
    grad_b = np.mean(error)
    W = W - lr * grad_W
    b = b - lr * grad_b
    return W,b
def model_V1(X, y, nf, epochs=500, lr=0.01):

    W, b = initialise(nf)
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = np.dot(xi, W) + b
            p = sigmoid(z)
            error = p - yi
            W, b = StochasticGradient(error, xi, W, b, lr)
    return W, b
def model_v2(X, y, nf, epochs=8000, lr=0.01):
    W, b = initialise(nf)
    for _ in range(epochs):
        error = logistic_regression(X,W,b,y)
        W, b = BatchGradientDescent(error, X, W, b, lr)
    return W, b
def model_V3(X, y,X_test,y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("Sklearn Logistic Accuracy:", acc)

def model_v4(X, y, epochs=15000, lr=0.005, lambda_=0.01):
    n_samples, n_features = X.shape
    W, b = initialise(n_features)
    for epoch in range(epochs):
        error,p = logistic_regression(X,W,b,y)
        W, b = BatchGradientDescentV2(error, X, W, b, lr,n_samples,lambda_)
        if epoch % 2000 == 0:
            loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
            print("Epoch:", epoch, "Loss:", loss)
    return W, b