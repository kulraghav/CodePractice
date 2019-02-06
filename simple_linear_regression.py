"""
    Simple Linear Regression
    
    Tasks:
    1. Generate samples [y = w.x + b + noise]
        Input: w, b, num_samples, low, high
        Output: num_samples 
    2. Fit simple linear regression to estimate w and b
    3. Evaluate r^2/R^2
     

Learning: low = 0, high = 1000 did not converge, learning_rate had to be kept very low
"""
import numpy as np
def generate_samples(w, b, num_samples, low=0, high=10, mu=0, sigma=1):
    samples = []
    for i in range(num_samples):
        x = np.random.uniform(low, high)
        eps = np.random.normal(mu, sigma)
        samples.append((x, w*x + b + eps))
    return samples

def get_gradient(w, b, sample):
    x = sample[0]
    y = sample[1]
    grad_w = 2*(w*x + b - y)*x 
    grad_b = 2*(w*x + b - y)*1
        
    return grad_w, grad_b    

def shuffle(samples):
    """
        randomly shuffle the order of samples
    """
    np.random.shuffle(samples)
    return samples

def predict(w, b, samples):
    y_pred = [w*sample[0] + b for sample in samples]
    return y_pred


def sgd_update(w, b, samples, learning_rate=0.01):
    samples = shuffle(samples)
    y = [sample[1] for sample in samples]
    y_pred = predict(w, b, samples)

    for sample in samples:
        grad_w, grad_b = get_gradient(w, b, sample)
        w = w - learning_rate*grad_w
        b = b - learning_rate*grad_b
        print(w, b)
        break

    return w, b

def initialize(samples):
    w = 0
    b = 0
    return w, b

def fit(samples, n_epochs=100, learning_rate=0.01, w_init=0, b_init=0, verbose=False):
    w = w_init
    b = b_init
    #w, b = initialize(samples)
    print(w, b)
    for i in range(n_epochs):
        w, b = sgd_update(w, b, samples, learning_rate)
        if verbose:
            print("(w, b) : {} mse : {}".format((w, b), mse_loss([w*sample[0] + b for sample in samples], [sample[1] for sample in samples])))
    return w, b

def evaluate(y_pred, y):
    y_avg = sum(y)/len(y)
    TSS = sum([(y - y_avg)**2])
    RSS = 0
    for i in range(len(y)):
        RSS = RSS + (y[i] - y_pred[i])**2
    R2 = 1 - RSS/TSS
    return R2
    
def mse_loss(y_pred, y):
    if not y:
        return 0
    TSS = 0
    for i in range(len(y)):
        TSS = sse + (y_pred[i]-y[i])**2
    mse = TSS / len(y)
    return mse
        

        
        
