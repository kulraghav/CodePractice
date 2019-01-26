"""
    Implement linear regression from scratch

    Variable | Dimension
    X        | n x p
    y        | n x 1
    W        | 1 x n
    b        | 1 x 1
    error    | n x 1
"""
import numpy as np

def predict(W, b, X):
    print("W.X")
    print(np.matmul(W, X))
    print("b")
    print(b)
    return np.dot(W, X) + b

def loss(y_pred, y):
    error = np.subtract(y, y_pred)
    return np.dot(error, error)/2*len(y)

def grad_W(X, error):
    return np.matmul(X, error)

def grad_b(X, error):
    return error

def update_weights(W, b, X, error, learning_rate):    
    W = W - learning_rate*grad_W(X, error)
    b = b - grad_b(X, error)
    return W, b

def train(X, y, num_iterations=100, learning_rate=0.1):
    W = np.array([0.0]).reshape(1, 1)
    b = 0.0

    for i in range(num_iterations):
        y_pred = predict(W, b, X)
        
        print (y_pred)
        break

        error = y - y_pred
        W, b = update_weights(W, b, X, error, learning_rate)

    return W, b


def unit_test():
   n = 1000
   p = 1
   A = list(range(1000))
   X = np.array(A).reshape(n, p)
   y = np.array([2*x + 3 for x in X]).reshape(n, 1)
   print train(X, y)
        
    
