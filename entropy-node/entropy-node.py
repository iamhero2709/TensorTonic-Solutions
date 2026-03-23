import numpy as np

def entropy_node(y):
    y = np.array(y)
    
    values, counts = np.unique(y, return_counts=True)
    p_i = counts / counts.sum()
    
    # Only take non-zero probabilities
    p_i = p_i[p_i > 0]
    
    entropy = -np.sum(p_i * np.log2(p_i))
    
    return float(entropy)