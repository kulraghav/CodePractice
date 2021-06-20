"""
    Multiple Linear Regression
    
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

def generate_samples(w, b, num_samples, low=0, high=1, mu=0, sigma=0.001):
    p = len(w)
    samples = []
    for i in range(num_samples):
        x = np.random.uniform(low, high, size=p)
        eps = np.random.normal(mu, sigma)
        samples.append((x, inner_product(w, x) + b + eps))
    return samples

def get_gradient(w, b, sample):
    x = sample[0]
    y = sample[1]

    y_pred = inner_product(w, x) + b 
    
    grad_w = [2*(y_pred - y)*e_x for e_x in x] 
    grad_b = 2*(y_pred - y)*1
        
    return grad_w, grad_b

def shuffle(samples):
    """
        randomly shuffle the order of samples
    """
    np.random.shuffle(samples)
    return samples

def predict(w, b, samples):
    y_pred = [inner_product(w, sample[0]) + b for sample in samples]
    return y_pred

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
            print("(w, b) : {} mse : {}".format((w, b), mse_loss(predict(w, b, samples), [sample[1] for sample in samples])))
    return w, b

def evaluate(y_pred, y):
    y_avg = sum(y)/len(y)
    TSS = 0
    RSS = 0
    for i in range(len(y)):
        RSS = RSS + (y[i] - y_pred[i])**2
        TSS = TSS + (y[i] - y_avg)**2
        # print((y[i] - y_pred[i])**2, (y[i] - y_avg)**2)
        
    R2 = 1 - (RSS/TSS)
    # print(list(zip(y_pred[:10], y[:10])))
    return R2

def mse_loss(y_pred, y):
    if not y:
        return 0
    TSS = 0
    for i in range(len(y)):
        TSS = TSS + (y_pred[i]-y[i])**2
    mse = TSS / len(y)
    return mse

"""
    learning: shuffle will change samples inside the function fit
    Earlier y = [sample[1] for sample in samples] was before the fit function and y_pred was after the fit function. 
    This was causing R^2 to be negative.
    Once I moved y = ... after the shuffle this issue was fixed.
"""
def test():
    samples = generate_samples([0.2, 0.2], 0.2, 100)
    w, b = fit(samples, n_epochs=100, verbose=True, learning_rate=0.001)
    y_pred = predict(w, b, samples)
    y = [sample[1] for sample in samples]
    R2 = evaluate(y_pred, y)
    print("R^2 : {}".format(R2))
    return R2
