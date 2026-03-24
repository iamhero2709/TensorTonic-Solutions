import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    
    def gini(y):
        # empty node → impurity 0
        if len(y) == 0:
            return 0.0
        
        y = np.array(y)
        
        # count each class
        _, counts = np.unique(y, return_counts=True)
        
        probs = counts / len(y)
        
        return 1.0 - np.sum(probs ** 2)
    
    
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    # if no samples at all
    if n_total == 0:
        return 0.0
    
    g_left = gini(y_left)
    g_right = gini(y_right)
    
    # weighted gini
    return (n_left / n_total) * g_left + (n_right / n_total) * g_right