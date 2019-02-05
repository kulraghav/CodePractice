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

def fit(samples):

    return w, b
        

        
        
