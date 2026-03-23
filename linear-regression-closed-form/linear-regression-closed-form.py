import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    x=np.array(X)
    y=np.array(y)
    w=np.linalg.inv(np.dot(x.T,x))@x.T@y
    return w 
    pass