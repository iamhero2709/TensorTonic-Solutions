import numpy as np

def random_forest_vote(predictions):
    predictions = np.array(predictions)
    
    result = []
    
    for col in predictions.T:
        values, counts = np.unique(col, return_counts=True)
        max_count = counts.max()
        
        # tie case → smallest label
        result.append(int(values[counts == max_count].min()))
    
    return result   # ⚠️ return list, not numpy array