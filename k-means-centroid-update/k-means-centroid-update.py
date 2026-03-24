def k_means_centroid_update(points, assignments, k):
    
    d = len(points[0])   # dimension
    
    # Step 1: initialize
    sums = []
    for i in range(k):
        sums.append([0.0] * d)
    
    counts = []
    for i in range(k):
        counts.append(0)
    
    # Step 2: add points manually
    for i in range(len(points)):
        point = points[i]
        cluster_id = assignments[i]
        
        for j in range(d):
            sums[cluster_id][j] = sums[cluster_id][j] + point[j]
        
        counts[cluster_id] = counts[cluster_id] + 1
    
    # Step 3: compute centroids
    centroids = []
    
    for i in range(k):
        centroid = []
        
        if counts[i] == 0:
            for j in range(d):
                centroid.append(0.0)
        else:
            for j in range(d):
                value = sums[i][j] / counts[i]
                centroid.append(value)
        
        centroids.append(centroid)
    
    return centroids