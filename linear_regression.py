"""
    Implement linear regression from scratch

    Variable | Dimension
    X        | n x p
    y        | n x 1
    W        | n x 1
    b        | 1 x 1
    error    | n x 1
"""

def predict(W, b, X):
    return np.add(np.dot(W, X), b)

def loss(y_pred, y):
    error = np.subtract(y, y_pred)
    return np.dot(error, error)/2*len(y)

def gradients(X, error):
    return np.matmul(X, error)

    
def train(X, y):
    pass
