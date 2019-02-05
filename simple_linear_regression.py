"""
    Simple Linear Regression
    
    Tasks:
    1. Generate samples [y = w.x + b + noise]
        Input: w, b, num_samples, low, high
        Output: num_samples 
    2. Fit simple linear regression to estimate w and b
    3. Evaluate r^2/R^2
     
"""
import numpy as np
def generate_samples(w, b, num_samples, low=0, high=1000, mu=0, sigma=1):
    samples = []
    for i in range(num_samples):
        x = np.random.uniform(low, high)
        eps = np.random.normal(mu, sigma)
        samples.append((x, w*x + b + eps))
    return samples

def mse_loss(y_pred, y):
    if not y:
        return 0
    sse = 0
    for i in range(len(y)):
        sse = sse + (y_pred[i]-y[i])**2
    mse = sse / len(y)
    return mse

def get_gradients(mse, samples):
    grad_w = 0
    grad_b = 0
    
    for sample in samples:
        x = sample[0]
        y = sample[1]
        grad_w = grad_w + 2*(w*x + b - y)*x 
        grad_b = grad_b + 2*(w*x + b - y)*1
    return grad_w, grad_b    

def sgd_update(w, b, samples, learning_rate=0.1):
    y = [sample[1] for sample in samples]
    y_pred = [w*sample[0] + b for sample in samples]
    mse = mse_loss(y_pred, y)

    grad_w, grad_b = get_gradients(mse, samples)

    w = w - learning_rate*grad_w
    b = b - learning_rate*grad_b

    return w, b
    
def fit(samples, num_iterations):
    w, b = initialize(samples)

    for i in range(num_iterations):
        w, b = sgd_update(w, b, samples)
        
    return w, b
        

        
        
