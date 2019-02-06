"""
    Logistic Regression
    
    Tasks:
    1. Generate samples [y = w.x + b + noise]
        Input: w, b, num_samples, low, high
        Output: num_samples 
    2. Fit simple linear regression to estimate w and b
    3. Evaluate R^2
"""
import numpy as np

def inner_product(w, x):
    ip = 0
    for i in range(len(w)):
      ip = ip + w[i]*x[i]
    return ip

from math import exp
def sigmoid(z):
    return exp(z)/(exp(-z) + exp(z))

def generate_samples(w, b, num_samples, low=0, high=1, mu=0, sigma=0.0001):
    p = len(w)
    samples = []
    for i in range(num_samples):
        x = np.random.uniform(low, high, size=p)
        y = chop(sigmoid(inner_product(w, x) + b))
        eps = np.random.normal(mu, sigma)
        samples.append((x, y))
    return samples

def get_gradient(w, b, sample):
    x = sample[0]
    y = sample[1]

    y_proba = sigmoid(inner_product(w, x) + b) 
    
    grad_w = [((1-y)*(1-y_proba) - y*y_proba)*e_x for e_x in x] 
    grad_b = ((1-y)*(1-y_proba) - y*y_proba))*1
        
    return grad_w, grad_b

def shuffle(samples):
    """
        randomly shuffle the order of samples
    """
    np.random.shuffle(samples)
    return samples

def chop(s, threshold=0.5):
    if s > 0.5:
        return 1
    else:
        return 0
    

def predict(w, b, samples):
    y_pred = [chop(sigmoid(inner_product(w, sample[0]) + b)) for sample in samples]
    return y_pred

def predict_probas(w, b, samples):
    y_probas = [sigmoid(inner_product(w, sample[0]) + b) for sample in samples]
    return y_probas

def sgd_update(w, b, samples, learning_rate=0.01):
    samples = shuffle(samples)
    y = [sample[1] for sample in samples]
    y_pred = predict(w, b, samples)

    for sample in samples:
        grad_w, grad_b = get_gradient(w, b, sample)
        for i in range(len(w)):
            w[i] = w[i] - learning_rate*grad_w[i]
        b = b - learning_rate*grad_b
    return w, b

def initialize(samples):
    p = len(samples[0][0])
    w = [0]*p
    b = 0
    return w, b

"""
    w_init: len of w_init has to be fixed, it is p = number of features
"""
def fit(samples, n_epochs=300, learning_rate=0.01, w_init=[0, 0], b_init=0, verbose=False):
    w = w_init
    b = b_init
    #w, b = initialize(samples)
    print(w, b)
    for i in range(n_epochs):
        w, b = sgd_update(w, b, samples, learning_rate)
        if verbose:
            print("(w, b) : {} accuracy : {}".format((w, b), accuracy(predict(w, b, samples), [sample[1] for sample in samples])))
    return w, b

def evaluate(y_pred, y):
    return accuracy(y_pred, y)

def accuracy(y_pred, y):
    count = 0
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            count = count + 1
    return count

def test():
    samples = generate_samples([0.2, 0.2], 0.2, 100)
    print(samples[:10])
    y = [sample[1] for sample in samples]
    w, b = fit(samples, n_epochs=100, verbose=True, learning_rate=0.001)
    y_pred = predict(w, b, samples)
    return  evaluate(y_pred, y)
    
