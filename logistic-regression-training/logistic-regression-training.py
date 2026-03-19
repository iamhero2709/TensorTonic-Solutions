import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    
    # ✅ ensure float arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    n_sample, n_feature = X.shape
    
    w = np.zeros(n_feature)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)

        error = y_pred - y

        dw = np.dot(X.T, error) / n_sample
        db = np.sum(error) / n_sample

        w -= lr * dw
        b -= lr * db

    return w, b